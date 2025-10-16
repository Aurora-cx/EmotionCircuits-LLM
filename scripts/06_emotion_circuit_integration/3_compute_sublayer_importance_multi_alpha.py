# -*- coding: utf-8 -*-
# scripts/06_emotion_circuit_integration/3_compute_sublayer_importance_multi_alpha.py
"""
计算每个"子层(Attention/MLP)×层"的单位重要性 I_{L,p}
Compute unit importance I_{L,p} for each "sublayer(Attention/MLP)×layer"

I_{L,p} = Δs / (α * σ_{L,p}),  Δs = <h_final' - h_final, v_ref^(emotion)>
- σ_{L,p} 来自 06_emotion_circuit_integration/sigma_summary.json
- σ_{L,p} from 06_emotion_circuit_integration/sigma_summary.json
- v_ref^(emotion) 用稳定段 L21–25 的 attn+mlp 合并平均（单位化、符号对齐）
- v_ref^(emotion) uses stable segment L21–25 attn+mlp merged average (normalized, sign-aligned)
- 注入口径：子层"输出→残差"那一支，仅 last token
- Injection scope: sublayer "output→residual" branch, last token only
- 评估口径：默认 final(L27)；可同时输出 anchor(L21–25 平均) 的投影变化
- Evaluation scope: default final(L27); optionally output anchor(L21–25 average) projection changes
- 支持多个alpha值计算，避免alpha带来的偶然性
- Support multiple alpha values computation to avoid alpha-induced randomness

- 输入 Input: outputs/{model_name}/02_emotion_directions/emo_directions_*.pt
- 输出 Output: outputs/{model_name}/06_emotion_circuit_integration/sublayer_importance/
"""

import os, json, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============== 路径与基本配置 / Paths and Basic Configuration ==============
# 工作目录：项目根目录
# Working directory: project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "llama32_3b"
EMOTIONS = ["anger","sadness","happiness","fear","surprise","disgust"]
N_LAYERS = 28
H = 3072

# ============== 工具函数 / Utility Functions ==============
def seed_all(seed:int):
    """
    设置随机种子
    Set random seeds
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    """
    确保目录存在
    Ensure directory exists
    """
    os.makedirs(p, exist_ok=True); return p

def build_messages_no_emo(scenario, event):
    """
    构建无情绪的消息
    Build messages without emotion
    """
    system = 'Keep the reply to at most two sentences.'
    user   = f"{scenario}\n{event}"
    return [{"role":"system","content":system},{"role":"user","content":user}]

def apply_chat_template(tok, scenario, event):
    """
    应用聊天模板
    Apply chat template
    """
    messages = build_messages_no_emo(scenario, event)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def l2_normalize(x:torch.Tensor, eps=1e-12):
    """
    L2归一化
    L2 normalization
    """
    return x / (x.norm(p=2) + eps)

def sign_align(v:torch.Tensor, ref:torch.Tensor):
    """
    符号对齐
    Sign alignment
    """
    return v if torch.dot(v, ref) >= 0 else -v

# ============== 数据加载器 / Data Loaders ==============
def load_sigma(path_02):
    """
    加载sigma值
    Load sigma values
    """
    p = os.path.join(path_02, "sigma_summary.json")
    with open(p, "r", encoding="utf-8") as f:
        sigma = json.load(f)
    # dict: {"attention_0": float, "mlp_0": float, ...}
    return sigma

def _load_dirs_pt(pt_path):
    """
    兼容 02_emotion_directions 下存的情绪方向文件格式
    Compatible with emotion direction file format under 02_emotion_directions
    
    pt['dirs'][emotion] -> list of per-layer vectors (numpy or list), shape (H,) each
    返回：dirs: {emo: [L -> np.ndarray(H,)]}
    Returns: dirs: {emo: [L -> np.ndarray(H,)]}
    """
    obj = torch.load(pt_path, map_location="cpu")
    if "dirs" not in obj or "emotions" not in obj:
        raise ValueError(f"Invalid dirs file: {pt_path}")
    out = {}
    for e in obj["emotions"]:
        seq = []
        for v in obj["dirs"][e]:
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape[0] != H:
                raise ValueError(f"Dim mismatch in {pt_path}, emotion={e}, got {arr.shape}")
            # 单位化（稳）
            # Normalization (stable)
            n = np.linalg.norm(arr) + 1e-12
            seq.append((arr / n).astype(np.float32))
        # length N_LAYERS
        out[e] = seq
    return out, obj.get("layers", N_LAYERS)

def try_load_sublayer_dirs(dir_02):
    """
    期望文件
    Expected files:
      - 02_emotion_directions/emo_directions_attention.pt  -> attention 子层情绪向量
      - 02_emotion_directions/emo_directions_attention.pt  -> attention sublayer emotion vectors
      - 02_emotion_directions/emo_directions_mlp.pt   -> mlp 子层情绪向量
      - 02_emotion_directions/emo_directions_mlp.pt   -> mlp sublayer emotion vectors
    
    若缺失某个则跳过该子层
    Skip sublayer if missing
    
    返回
    Returns:
      dirs = { "attention": {emo: [L->np(H,)]}, "mlp": {emo: [L->np(H,)]} }
    """
    out = {}
    attn_p = os.path.join(dir_02, "emo_directions_attention.pt")
    mlp_p  = os.path.join(dir_02, "emo_directions_mlp.pt")
    found = False
    if os.path.exists(attn_p):
        dirs_attn, nL = _load_dirs_pt(attn_p)
        out["attention"] = dirs_attn; found = True
    if os.path.exists(mlp_p):
        dirs_mlp, nL2 = _load_dirs_pt(mlp_p)
        out["mlp"] = dirs_mlp; found = True
    if not found:
        raise FileNotFoundError(
            f"Not found sublayer direction files under {dir_02}. "
            f"Expect emo_directions_attention.pt and/or emo_directions_mlp.pt"
        )
    return out

def load_accepted_samples(accepted_file):
    """
    加载accepted样本
    Load accepted samples
    """
    rows = []
    if not os.path.exists(accepted_file): return rows
    with open(accepted_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def expand_accepted_samples(accepted_rows, emotions):
    """
    将accepted样本按情绪展开
    Expand accepted samples by emotion
    """
    samples = []
    for r in accepted_rows:
        scenario = r.get("scenario","")
        event = r.get("event","")
        skeleton = r.get("skeleton_id","")
        valence = r.get("valence","")
        emotion = r.get("emotion","")
        
        # 只保留指定情绪的样本
        # Only keep samples with specified emotions
        if emotion in emotions:
            samples.append(dict(
                scenario=scenario, event=event, emotion=emotion,
                skeleton_id=skeleton, valence=valence
            ))
    return samples

# ============== 参考基构建器 / Reference Builder ==============
def build_global_reference(dirs_by_sublayer, emotions, stable_layers=(21,22,23,24,25)):
    """
    在稳定段 L∈{21..25}，把 attention+mlp 的单位向量合并平均，得到每个情绪的统一参考基 v_ref^(e)（单位化）
    In stable segment L∈{21..25}, merge and average attention+mlp unit vectors to get unified reference base v_ref^(e) for each emotion (normalized)
    """
    vref = {}
    for e in emotions:
        vecs = []
        # "attention"/"mlp"
        for p in dirs_by_sublayer.keys():
            # [L -> np(H,)]
            seq = dirs_by_sublayer[p][e]
            for L in stable_layers:
                if L < len(seq):
                    v = torch.tensor(seq[L], dtype=torch.float32)
                    vecs.append(l2_normalize(v))
        if not vecs:
            raise ValueError(f"No vectors collected for emotion={e} in stable segment")
        # 符号对齐 + 平均
        # Sign alignment + average
        ref = vecs[0].clone()
        acc = torch.zeros_like(ref)
        for v in vecs:
            acc += sign_align(v, ref)
        # {emo: np(H,)}
        vref[e] = l2_normalize(acc).cpu().numpy()
    return vref

# ============== 注入钩子 / Injection Hooks ==============
class SublayerInjector:
    """
    在指定层/子层对 last token 注入 Δh（与子层输出残差对齐的口径）
    Inject Δh to last token at specified layer/sublayer (aligned with sublayer output residual scope)
    
      - attention: 注册在 layer.self_attn.forward 输出 (B,T,H) 上加 Δh 到最后 token
      - attention: Register on layer.self_attn.forward output (B,T,H) add Δh to last token
      - mlp      : 注册在 layer.mlp.forward      输出 (B,T,H) 上加 Δh 到最后 token
      - mlp      : Register on layer.mlp.forward output (B,T,H) add Δh to last token
    """
    def __init__(self, plan):
        # list of dicts: {layer, kind, delta_vec: torch(H,)}
        self.plan = plan
        self.handles = []

    def _hook_attn(self, delta_vec):
        def fn(module, inputs, output):
            # output: tuple of (hidden_states, attention_weights, ...)
            if isinstance(output, tuple):
                # (B, T, H)
                hidden_states = output[0]
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + delta_vec
                # 保持其他输出不变
                # Keep other outputs unchanged
                return (hidden_states,) + output[1:]
            else:
                # output: (B, T, H)
                output[:, -1, :] = output[:, -1, :] + delta_vec
                return output
        return fn

    def _hook_mlp(self, delta_vec):
        def fn(module, inputs, output):
            # output: (B, T, H)
            output[:, -1, :] = output[:, -1, :] + delta_vec
            return output
        return fn

    def register(self, model):
        for item in self.plan:
            L = item["layer"]; kind = item["kind"]; dv = item["delta_vec"]
            layer = model.model.layers[L]
            if kind == "attention":
                h = layer.self_attn.register_forward_hook(self._hook_attn(dv))
            elif kind == "mlp":
                h = layer.mlp.register_forward_hook(self._hook_mlp(dv))
            else:
                raise ValueError(f"Unknown kind={kind}")
            self.handles.append(h)

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

# ============== 核心计算 / Core Computation ==============
def compute_importance_per_sublayer(
    model, tok, samples, emotion, dirs_by_sublayer, v_ref,
    sigma_map, alpha=0.5, anchor_mode="final", device="cpu",
    stable_layers=(21,22,23,24,25)
):
    """
    返回 DataFrame: 每个 (layer, kind) 的单位重要性
    Returns DataFrame: unit importance for each (layer, kind)
    
    columns: [emotion, kind, layer, alpha, sigma, dose, s_base_mean, ds_mean_final, I_final, ds_mean_anchor?, I_anchor?]
    """
    # 预先把本情绪的方向张量化
    # Pre-tensorize emotion direction
    # (H,)
    vref = torch.tensor(v_ref[emotion], dtype=torch.float32, device=device)
    
    # 建立子层本地向量缓存（单位化+与 vref 符号对齐）
    # Build sublayer local vector cache (normalized + sign-aligned with vref)
    # {(kind, L): torch(H,)}
    local_dirs = {}
    # e.g. ["attention","mlp"]
    kinds = list(dirs_by_sublayer.keys())
    for kind in kinds:
        for L in range(N_LAYERS):
            vec = torch.tensor(dirs_by_sublayer[kind][emotion][L], dtype=torch.float32, device=device)
            vec = l2_normalize(vec)
            vec = sign_align(vec, vref)
            local_dirs[(kind, L)] = vec

    # baseline：对所有样本跑一遍无注入，取 final(L27) 与 anchor 的投影
    # baseline: run all samples without injection, get final(L27) and anchor projections
    s_base_final = []
    s_base_anchor = []
    for r in tqdm(samples, desc=f"[{emotion}] baseline", leave=False):
        prompt = apply_chat_template(tok, r.get("scenario",""), r.get("event",""))
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False, output_hidden_states=True)
            # final
            # (B, T, H)
            h_final = out.hidden_states[-1]
            s_f = torch.matmul(h_final[:, -1, :], vref).item()
            s_base_final.append(s_f)
            # anchor: L21–25 平均（可选）
            # anchor: L21–25 average (optional)
            if anchor_mode == "anchor+final":
                acc = 0.0
                for L in stable_layers:
                    # HF: hidden_states[0]=embeds, layer L -> index L+1
                    hL = out.hidden_states[L+1]
                    acc += torch.matmul(hL[:, -1, :], vref).item()
                s_base_anchor.append(acc / len(stable_layers))

    base_final_mean = float(np.mean(s_base_final))
    base_anchor_mean = float(np.mean(s_base_anchor)) if s_base_anchor else None

    rows = []

    # 逐 (kind, L) 计算 Δs 与单位重要性
    # Compute Δs and unit importance for each (kind, L)
    total_layers = len(kinds) * N_LAYERS
    current_layer = 0
    
    for kind in kinds:
        print(f"\n=== Processing {kind} layers ===")
        for L in range(N_LAYERS):
            current_layer += 1
            key = f"{kind}_{L}"
            # 缺 σ 则跳过
            # Skip if σ missing
            if key not in sigma_map:
                continue
            sigma = float(sigma_map[key])
            # 剂量：α * σ
            # Dose: α * σ
            dose = alpha * sigma

            # 组装注入计划：仅本子层、仅 last token、Δh = dose * v_local
            # Assemble injection plan: only this sublayer, only last token, Δh = dose * v_local
            v_local = local_dirs[(kind, L)]
            # (1,1,H) 便于广播
            # (1,1,H) for broadcasting
            delta_vec = (dose * v_local).unsqueeze(0).unsqueeze(0)
            plan = [dict(layer=L, kind=kind, delta_vec=delta_vec)]

            injector = SublayerInjector(plan)
            injector.register(model)

            try:
                s_fin_list = []
                s_anc_list = []
                for r in tqdm(samples, desc=f"[{emotion}] {kind}@L{L}", leave=False):
                    prompt = apply_chat_template(tok, r.get("scenario",""), r.get("event",""))
                    inputs = tok(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model(**inputs, use_cache=False, output_hidden_states=True)
                        # final
                        h_final = out.hidden_states[-1]
                        s_f = torch.matmul(h_final[:, -1, :], vref).item()
                        s_fin_list.append(s_f)
                        # anchor
                        if anchor_mode == "anchor+final":
                            acc = 0.0
                            for La in stable_layers:
                                hL = out.hidden_states[La+1]
                                acc += torch.matmul(hL[:, -1, :], vref).item()
                            s_anc_list.append(acc / len(stable_layers))
            finally:
                injector.remove()

            # 统计
            # Statistics
            s_fin_mean = float(np.mean(s_fin_list))
            ds_final   = s_fin_mean - base_final_mean
            I_final    = ds_final / max(dose, 1e-12)

            if anchor_mode == "anchor+final":
                s_anc_mean = float(np.mean(s_anc_list))
                ds_anchor  = s_anc_mean - base_anchor_mean
                I_anchor   = ds_anchor / max(dose, 1e-12)
            else:
                s_anc_mean = None; ds_anchor = None; I_anchor = None

            # 输出当前层的结果
            # Output current layer results
            print(f"  [{emotion}] {kind}@L{L:2d} ({current_layer}/{total_layers}): σ={sigma:.6f}, dose={dose:.6f}")
            print(f"    baseline_final={base_final_mean:.6f}, final={s_fin_mean:.6f}, Δs={ds_final:.6f}, I={I_final:.6f}")
            if anchor_mode == "anchor+final":
                print(f"    baseline_anchor={base_anchor_mean:.6f}, anchor={s_anc_mean:.6f}, Δs={ds_anchor:.6f}, I={I_anchor:.6f}")
            # 空行分隔
            # Empty line separator
            print()

            rows.append(dict(
                emotion=emotion, kind=kind, layer=L,
                alpha=float(alpha), sigma=float(sigma), dose=float(dose),
                s_base_final=base_final_mean, s_final=s_fin_mean, ds_final=ds_final, I_final=I_final,
                s_base_anchor=s_base_anchor if base_anchor_mean is not None else None,
                s_anchor=s_anc_mean, ds_anchor=ds_anchor, I_anchor=I_anchor,
                n_samples=len(samples)
            ))

    df = pd.DataFrame(rows)
    return df

# ============== 主函数 / Main Function ==============
def main():
    """
    主函数
    Main function
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=MODEL_NAME, 
                    help="模型名称 / Model name")
    ap.add_argument("--emotions", default="anger,sadness,happiness,fear,disgust,surprise", 
                    help="情绪列表 / Emotion list")
    ap.add_argument("--alphas", default="0.1,0.3,0.5,0.7", 
                    help="多个alpha值，用逗号分隔 / Multiple alpha values, comma-separated")
    ap.add_argument("--anchor_mode", default="final", choices=["final","anchor+final"],
                    help="是否同时输出稳定段(L21–25)均值投影的Δs与单位效应 / Whether to output stable segment(L21–25) mean projection Δs and unit effects")
    ap.add_argument("--seed", type=int, default=42, 
                    help="随机种子 / Random seed")
    args = ap.parse_args()

    seed_all(args.seed)

    # 解析alpha值
    # Parse alpha values
    alpha_values = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    print(f"[+] Alpha values: {alpha_values}")

    # 路径
    # Paths
    base_dir = PROJECT_ROOT / "outputs" / args.model_name
    dir_02 = base_dir / "02_emotion_directions"
    dir_01 = base_dir / "01_emotion_elicited_generation_prompt_based"
    dir_06 = ensure_dir(base_dir / "06_emotion_circuit_integration")
    dir_out = ensure_dir(dir_06 / "sublayer_importance")

    # 加载 σ
    # Load σ
    sigma_map = load_sigma(str(dir_06))

    # 加载子层情绪向量（attention / mlp）
    # Load sublayer emotion vectors (attention / mlp)
    # {"attention":{emo:[L->np(H,)]}, "mlp":{...}}
    dirs_by_sublayer = try_load_sublayer_dirs(str(dir_02))

    # 构建统一参考基（L21–25，attn+mlp 合并）
    # Build unified reference base (L21–25, attn+mlp merged)
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    v_ref = build_global_reference(dirs_by_sublayer, emotions, stable_layers=(21,22,23,24,25))

    # 样本 - 使用accepted样本
    # Samples - use accepted samples
    accepted_file = dir_01 / "labeled" / "sev" / "accepted.jsonl"
    accepted_rows = load_accepted_samples(str(accepted_file))
    if not accepted_rows:
        print("[-] No accepted samples found."); return
    samples = expand_accepted_samples(accepted_rows, emotions)
    print(f"[+] Accepted samples loaded: {len(samples)}")

    # 模型
    # Model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # HuggingFace token处理
    # HuggingFace token handling
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    
    # 加载模型和tokenizer
    # Load model and tokenizer
    print(f"[+] Loading model: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, token=HF_TOKEN if HF_TOKEN else True)
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map={"": device}, token=HF_TOKEN if HF_TOKEN else True
    )
    print(f"[+] Model loaded on device: {device}")
    model.eval()

    # 按alpha值分别计算
    # Compute separately by alpha values
    all_alpha_results = []
    
    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"Computing importance with alpha = {alpha}")
        print(f"{'='*60}")
        
        # 按情绪分别计算
        # Compute separately by emotions
        all_rows = []
        for emo in emotions:
            print(f"\n=== Computing sublayer importance for emotion: {emo} (alpha={alpha}) ===")
            df = compute_importance_per_sublayer(
                model, tok, [s for s in samples if s["emotion"]==emo],
                emotion=emo,
                dirs_by_sublayer=dirs_by_sublayer,
                v_ref=v_ref,
                sigma_map=sigma_map,
                alpha=alpha,
                anchor_mode=args.anchor_mode,
                device=device,
                stable_layers=(21,22,23,24,25)
            )
            out_csv = dir_out / f"importance_{emo}_alpha{alpha}.csv"
            df.to_csv(str(out_csv), index=False)
            print(f"[✓] Saved: {out_csv}")
            all_rows.append(df)

        # 合并一个总表
        # Merge into one summary table
        if all_rows:
            all_df = pd.concat(all_rows, ignore_index=True)
            all_csv = dir_out / f"importance_all_alpha{alpha}.csv"
            all_df.to_csv(str(all_csv), index=False)
            print(f"[✓] Saved: {all_csv}")
            all_alpha_results.append(all_df)

    # 合并所有alpha的结果
    # Merge all alpha results
    if all_alpha_results:
        combined_df = pd.concat(all_alpha_results, ignore_index=True)
        combined_csv = dir_out / "importance_all_alphas.csv"
        combined_df.to_csv(str(combined_csv), index=False)
        print(f"[✓] Saved combined results: {combined_csv}")

    # 同时保存参考基
    # Also save reference base
    ref_dir = ensure_dir(dir_06 / "global_ref")
    for e, v in v_ref.items():
        np.save(str(ref_dir / f"v_ref_{e}.npy"), v.astype(np.float32))
    print(f"[✓] Global reference saved to: {ref_dir}")

    print(f"\n[Done] Multi-alpha sublayer importance computed.")
    print(f"Results saved to: {dir_out}")

if __name__ == "__main__":
    main()
