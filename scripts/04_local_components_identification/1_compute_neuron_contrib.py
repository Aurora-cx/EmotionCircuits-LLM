# scripts/04_local_components_identification/1_compute_neuron_contrib.py
# -*- coding: utf-8 -*-
"""
神经元贡献计算脚本
Neuron Contribution Calculation Script

计算逐神经元对"把 hidden 沿情绪方向推进"的贡献
Calculate per-neuron contribution to pushing hidden states along emotion directions

方法 Method:
  beta = W_down @ v_emo[L]      (shape: d_ff)
  c^{(n)} = H^{(n)} * beta      (shape: d_ff), H 为 SwiGLU 门后激活的 last-token 行
  
输出 Output:
  1) 逐样本矩阵 Per-sample matrices: outputs/{model_name}/04_local_components_identification/mlp_neurons/per_sample/{emotion}/layer{L}.npz
       - c: (N, d_ff)
       - H: (N, d_ff)
       - beta: (d_ff,)
     以及 meta.json (与 c/H 的样本顺序对齐)
  2) 聚合均值表 Aggregated statistics: outputs/{model_name}/04_local_components_identification/mlp_neurons/contrib_mean_{emotion}.csv

依赖 Dependencies:
  - outputs/{model_name}/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl
  - outputs/{model_name}/02_emotion_directions/emo_directions_mlp.pt
"""

import os, json, argparse, random
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm

# HF token (优先使用环境变量，如果没有则尝试默认登录)
# HF token (prioritize env var, otherwise use default login)
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if HF_TOKEN:
    login(token=HF_TOKEN)

VALENCES = ["positive","neutral","negative"]

# ---------------- 工具函数 Utils ----------------
def seed_all(seed:int):
    """设置所有随机种子 Set all random seeds"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    """确保目录存在 Ensure directory exists"""
    os.makedirs(p, exist_ok=True); return p

def build_messages(emotion, scenario, event):
    """构建对话消息 Build chat messages"""
    system = f"Always reply in {emotion}.\nKeep the reply to at most two sentences."
    user   = f"{scenario}\n{event}"
    return [{"role":"system","content":system}, {"role":"user","content":user}]

def apply_chat_template(tok, emotion, scenario, event):
    """应用聊天模板 Apply chat template"""
    messages = build_messages(emotion, scenario, event)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_emo_dirs(model_name):
    """
    加载情绪方向向量
    Load emotion direction vectors
    """
    p = os.path.join("outputs", model_name, "02_emotion_directions", "emo_directions_mlp.pt")
    obj = torch.load(p, map_location="cpu")
    dirs = {e: obj["dirs"][e] for e in obj["emotions"]}  # {emo: [L,H]}
    n_layers = obj["layers"]; Hdim = obj["hidden"]
    # 单位化（稳妥）Normalize (for stability)
    for e in dirs:
        v = dirs[e]
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        dirs[e] = (v / n).astype(np.float32)
    return dirs, n_layers, Hdim, obj["emotions"]

def load_accepted_meta(model_name):
    """
    加载已接受的样本元数据
    Load accepted sample metadata
    """
    jsonl = os.path.join("outputs", model_name, "01_emotion_elicited_generation_prompt_based", 
                         "labeled", "sev", "accepted.jsonl")
    rows = []
    if not os.path.exists(jsonl):
        print(f"[-] Not found: {jsonl}")
        return rows
    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if bool(row.get("judge",{}).get("match",0)):
                rows.append(row)
    return rows

def stratified_pick(rows, emotion, max_per_valence=None):
    """
    按极性分层抽样
    Stratified sampling by valence
    """
    by_v = {v: [] for v in VALENCES}
    for r in rows:
        if r.get("emotion")==emotion:
            v = r.get("valence")
            if v in by_v: by_v[v].append(r)
    sel = []
    for v in VALENCES:
        L = by_v[v]
        if not L: continue
        if (max_per_valence is not None) and (len(L)>max_per_valence):
            sel.extend(random.sample(L, max_per_valence))
        else:
            sel.extend(L)
    return sel

# ---------------- Hook 类 Hook Classes ----------------
class MlpTap:
    """
    MLP层激活提取器
    MLP Layer Activation Extractor
    
    在 layer.mlp 的 forward hook 内提取：
    Extract from layer.mlp forward hook:
      mlp_input -> gate_proj/up_proj/act -> H = SiLU(gate) * up -> 取 last-token 行 (B, d_ff)
      记录 down_proj 以便主循环计算 beta = down_proj.T @ v
      Record down_proj for beta calculation in main loop
    """
    def __init__(self, layer_id, v_dir_vec_np):
        self.layer_id = layer_id
        self.v_np = v_dir_vec_np  # numpy [H] - 情绪方向向量 emotion direction vector
        self.H_last = None        # (B, d_ff) - last token激活 last token activation
        self.down_proj = None     # down投影层 down projection layer
        self.handle = None        # hook句柄 hook handle

    def mlp_hook(self, module, inputs, output):
        """
        MLP前向传播hook
        MLP forward hook
        """
        mlp_input = inputs[0]                 # (B, T, H)
        gate_out  = module.gate_proj(mlp_input)   # (B, T, d_ff)
        up_out    = module.up_proj(mlp_input)     # (B, T, d_ff)
        activated = module.act_fn(gate_out) * up_out   # SwiGLU 门后激活 SwiGLU gated activation
        a_last = activated[:, -1, :]          # (B, d_ff) 取 last-token Extract last token
        self.H_last = a_last
        self.down_proj = module.down_proj     # 记下本层 down_proj Record down_proj

    def register(self, model):
        """注册hook Register hook"""
        layer = model.model.layers[self.layer_id]
        self.handle = layer.mlp.register_forward_hook(self.mlp_hook)

    def remove(self):
        """移除hook Remove hook"""
        if self.handle is not None: self.handle.remove()

# ---------------- 核心计算 Core Computation ----------------
def compute_contrib_per_sample(model, tok, emotion, samples, layers, v_dirs):
    """
    计算每个样本的神经元贡献
    Compute neuron contributions per sample
    
    返回 Returns:
      per_layer: {L -> dict(H: np.ndarray[N,d_ff], c: np.ndarray[N,d_ff], beta: np.ndarray[d_ff])}
      meta_rows: 与 per-layer 的样本顺序对齐 Aligned with per-layer sample order
    """
    device = model.device
    taps = {L: MlpTap(L, v_dirs[emotion][L].astype(np.float32)) for L in layers}
    for t in taps.values(): t.register(model)

    per_layer = {L: {"H": [], "c": [], "beta": None} for L in layers}
    meta_rows = []

    model.eval()
    with torch.no_grad():
        for r in tqdm(samples, desc=f"[{emotion}] contributions", unit="sample"):
            prompt = apply_chat_template(tok, emotion, r.get("scenario",""), r.get("event",""))
            inputs = tok(prompt, return_tensors="pt").to(device)

            # 不生成，仅前向；禁用 cache 保持 hook 的简洁
            # No generation, forward only; disable cache for simpler hooks
            _ = model(**inputs, use_cache=False)

            for L in layers:
                tap = taps[L]
                H_last = tap.H_last
                dp = tap.down_proj
                if (H_last is None) or (dp is None):
                    continue

                # 计算 beta = W_down @ v
                # Calculate beta = W_down @ v
                # 注意 down_proj: [d_ff -> H]
                # Note down_proj: [d_ff -> H]
                # W_down 的 shape 是 (d_model, d_ff) (线性层权重通常是 out_features x in_features)
                # W_down shape is (d_model, d_ff) (linear layer weight is usually out_features x in_features)
                # HF里 Linear.forward: y = x @ W^T + b
                # In HF Linear.forward: y = x @ W^T + b
                # 对 down_proj(a): a:[B,d_ff], dp.weight:[d_model,d_ff] (out,in)
                # For down_proj(a): a:[B,d_ff], dp.weight:[d_model,d_ff] (out,in)
                # 我们需要 beta = W_down @ v (in_features 维度乘 v)
                # We need beta = W_down @ v (multiply along in_features dimension)
                # beta:[d_ff] = dp.weight.T @ v  其中 v:[d_model]
                # beta:[d_ff] = dp.weight.T @ v  where v:[d_model]
                W_down = dp.weight.detach()    # [d_model, d_ff]
                v = torch.tensor(tap.v_np, device=device, dtype=W_down.dtype)  # [H=d_model]
                beta_vec = (W_down.T @ v).contiguous()  # [d_ff]

                if per_layer[L]["beta"] is None:
                    per_layer[L]["beta"] = beta_vec.detach().to("cpu").float().numpy()

                # 本样本的 H 与 c
                # Current sample's H and c
                H_np = H_last[0].detach().to("cpu").float().numpy()   # (d_ff,)
                beta_np = per_layer[L]["beta"]                        # (d_ff,)
                c_np = H_np * beta_np                                 # (d_ff,)

                per_layer[L]["H"].append(H_np)
                per_layer[L]["c"].append(c_np)

            meta_rows.append({
                "key": r.get("key",""),
                "valence": r.get("valence",""),
                "skeleton_id": r.get("skeleton_id",""),
            })

    for t in taps.values(): t.remove()

    # 堆叠为矩阵
    # Stack into matrices
    for L in layers:
        if len(per_layer[L]["H"])==0:
            continue
        per_layer[L]["H"] = np.stack(per_layer[L]["H"], axis=0)   # (N, d_ff)
        per_layer[L]["c"] = np.stack(per_layer[L]["c"], axis=0)   # (N, d_ff)

    return per_layer, meta_rows

def save_per_sample_npz(base_out_dir, emotion, per_layer, meta_rows):
    """
    保存每个样本的贡献矩阵
    Save per-sample contribution matrices
    """
    per_emotion_dir = ensure_dir(os.path.join(base_out_dir, "per_sample", emotion))
    # 保存每层的矩阵
    # Save matrices for each layer
    for L, d in per_layer.items():
        if isinstance(d.get("H"), list) and len(d["H"])==0:
            continue
        if "H" not in d or "c" not in d or d["H"] is None or d["c"] is None:
            continue
        path = os.path.join(per_emotion_dir, f"layer{L}.npz")
        np.savez_compressed(path, H=d["H"], c=d["c"], beta=d["beta"])
    # 保存 meta
    # Save metadata
    with open(os.path.join(per_emotion_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, ensure_ascii=False, indent=2)
    return per_emotion_dir

def save_aggregated_csv(base_out_dir, emotion, per_layer):
    """
    保存聚合统计CSV
    Save aggregated statistics CSV
    """
    import pandas as pd
    rows = []
    for L, d in per_layer.items():
        H = d.get("H", None)
        C = d.get("c", None)
        beta = d.get("beta", None)
        if H is None or C is None or beta is None:
            continue
        # 聚合：贡献均值与符号分解
        # Aggregation: contribution mean and sign decomposition
        C_mean   = C.mean(axis=0)                        # (d_ff,)
        C_pos    = np.clip(C, a_min=0, a_max=None).mean(axis=0)
        C_neg    = np.clip(-C, a_min=0, a_max=None).mean(axis=0)
        H_mean   = H.mean(axis=0)                        # (d_ff,)
        dff = C_mean.shape[0]
        for j in range(dff):
            rows.append(dict(
                emotion=emotion,
                layer=L, neuron=j,
                C_mean=float(C_mean[j]),
                C_pos=float(C_pos[j]),
                C_neg=float(C_neg[j]),
                beta=float(beta[j]),
                H_mean=float(H_mean[j]),
            ))
    df = pd.DataFrame(rows)
    p = os.path.join(base_out_dir, f"contrib_mean_{emotion}.csv")
    df.to_csv(p, index=False)
    return p

# ---------------- 主函数 Main ----------------
def main():
    ap = argparse.ArgumentParser(description="计算神经元贡献 Compute neuron contributions")
    ap.add_argument("--model_name", type=str, default="llama32_3b",
                   help="模型文件夹名称 Model folder name")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                   help="模型名称 Model name")
    ap.add_argument("--emotions", default="anger,sadness,happiness,fear,surprise,disgust",
                   help="情绪列表（逗号分隔）Emotion list (comma-separated)")
    ap.add_argument("--layers", default="", 
                   help="层号（逗号分隔；空=全层）Layer numbers (comma-separated; empty=all layers)")
    ap.add_argument("--max_per_valence", type=int, default=None,
                   help="每种极性最多抽多少样本（None=全部）Max samples per valence (None=all)")
    ap.add_argument("--device", type=str, default="cuda:0",
                   help="设备 Device")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","bfloat16","float16"],
                   help="数据类型 Data type")
    ap.add_argument("--seed", type=int, default=42,
                   help="随机种子 Random seed")
    args = ap.parse_args()

    seed_all(args.seed)

    # 确定输出目录
    # Determine output directory
    output_dir = ensure_dir(os.path.join("outputs", args.model_name, "04_local_components_identification", "mlp_neurons"))

    # 加载情绪方向向量
    # Load emotion direction vectors
    emo_dirs, n_layers, Hdim, emo_list = load_emo_dirs(args.model_name)
    
    # 目标层集合：默认全层
    # Target layers: default all layers
    if args.layers.strip():
        target_layers = [int(x) for x in args.layers.split(",") if x.strip()!=""]
    else:
        target_layers = list(range(n_layers))
    print(f"[+] Target layers: {target_layers}")

    # 样本元数据
    # Sample metadata
    rows = load_accepted_meta(args.model_name)
    if not rows:
        print("[-] No accepted samples found."); return

    # 设置数据类型
    # Set data type
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    torch_dtype = dtype_map[args.dtype]

    # 加载模型和分词器
    # Load model and tokenizer
    device = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=HF_TOKEN if HF_TOKEN else True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch_dtype, 
        device_map={"": device},
        token=HF_TOKEN if HF_TOKEN else True
    )
    model.eval()

    # 处理每个情绪
    # Process each emotion
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    for emo in tqdm(emotions, desc="Emotions", unit="emo"):
        samples = stratified_pick(rows, emo, max_per_valence=args.max_per_valence)
        if len(samples)==0:
            print(f"[!] No samples for emotion={emo}"); continue
        print(f"[{emo}] N={len(samples)} samples (stratified by valence)")

        per_layer, meta_rows = compute_contrib_per_sample(
            model, tok, emo, samples, target_layers, emo_dirs
        )

        per_emotion_dir = save_per_sample_npz(output_dir, emo, per_layer, meta_rows)
        csv_path = save_aggregated_csv(output_dir, emo, per_layer)
        print(f"[{emo}] saved per-sample to: {per_emotion_dir}")
        print(f"[{emo}] saved aggregated csv: {csv_path}")

    print(f"\n[✓] Done! Check outputs/{args.model_name}/04_local_components_identification/mlp_neurons/")
    print(f"    - per_sample/{{emotion}}/layer{{L}}.npz")
    print(f"    - contrib_mean_{{emotion}}.csv")

if __name__ == "__main__":
    main()