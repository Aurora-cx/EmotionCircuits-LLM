# scripts/04_local_components_identification/2_compute_head_contrib.py
# -*- coding: utf-8 -*-
"""
注意力头重要性分析脚本
Attention Head Importance Analysis Script

核心功能 Core Functionality:
1. 计算情绪向量与MLP输入hidden states的投影作为参考值
   Compute emotion vector projections with MLP input hidden states as reference
2. 逐个屏蔽attention head，观察对残差流的影响
   Mask individual attention heads and observe impact on residual stream
3. 识别对情绪表达最重要的attention head
   Identify most important attention heads for emotion expression

输入 Input:
- outputs/{model_name}/02_emotion_directions/emo_directions_attention.pt
- outputs/{model_name}/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl

输出 Output:
- outputs/{model_name}/04_local_components_identification/attention_heads/head_importance_{emotion}.csv

使用示例 Usage Examples:
python scripts/04_local_components_identification/2_compute_head_contrib.py --emotions anger
python scripts/04_local_components_identification/2_compute_head_contrib.py --max_samples 30
python scripts/04_local_components_identification/2_compute_head_contrib.py --emotions happiness,fear --layers 11-20
"""

import os, json, argparse, random
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd

# HF token (优先使用环境变量，如果没有则尝试默认登录)
# HF token (prioritize env var, otherwise use default login)
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if HF_TOKEN:
    login(token=HF_TOKEN)

EMOTIONS = ["anger", "sadness", "happiness", "fear", "surprise", "disgust"]
VALENCES = ["positive", "neutral", "negative"]

# 工具函数 Utils
def seed_all(seed: int):
    """设置所有随机种子 Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    """确保目录存在 Ensure directory exists"""
    os.makedirs(p, exist_ok=True)
    return p

def build_messages(emotion: str, scenario: str, event: str):
    """
    构建情绪引导对话消息
    Build emotion-guided conversation messages
    """
    system = f"Always reply in {emotion}.\nKeep the reply to at most two sentences."
    user = f"{scenario}\n{event}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def apply_chat_template(tok, emotion, scenario, event):
    """应用聊天模板 Apply chat template"""
    messages = build_messages(emotion, scenario, event)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_emo_dirs(model_name):
    """
    加载情绪方向向量（attention）
    Load emotion direction vectors (attention)
    """
    p = os.path.join("outputs", model_name, "02_emotion_directions", "emo_directions_attention.pt")
    
    # 处理 PyTorch weights_only 警告
    # Handle PyTorch weights_only warning
    obj = torch.load(p, map_location="cpu", weights_only=False)
    
    dirs = {e: obj["dirs"][e] for e in obj["emotions"]}  # {emo: [L,H]}
    L = obj["layers"]
    H = obj["hidden"]
    # 归一化为单位向量
    # Normalize to unit vectors
    for e in dirs:
        v = dirs[e]
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        dirs[e] = (v / n).astype(np.float32)
    return dirs, L, H, obj["emotions"]

def load_accepted_samples(model_name, emotion, max_samples=50):
    """
    加载成功样本数据（all valence）
    Load successful sample data (all valence)
    """
    jsonl_path = os.path.join("outputs", model_name, "01_emotion_elicited_generation_prompt_based",
                              "labeled", "sev", "accepted.jsonl")
    samples = []
    if not os.path.exists(jsonl_path):
        return samples
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if (row.get("emotion") == emotion and 
                    bool(row.get("judge", {}).get("match", 0))):
                    samples.append(row)
                    if len(samples) >= max_samples:
                        break
            except Exception:
                continue
    return samples

# Hook类 Hook Classes
class HiddenStateCollector:
    """
    Hidden State收集器，收集MLP输入前的hidden states（即attention残差流之后）
    Collect hidden states before MLP input (i.e., after attention residual stream)
    """
    def __init__(self, target_layers):
        self.target_layers = set(target_layers)
        self.hidden_states = {}  # {layer_id: tensor}
        self.handles = []

    def _make_hook(self, layer_id):
        def hook(module, inputs, output):
            # inputs here are MLP inputs, i.e., hidden states after attention residual stream
            if isinstance(inputs, (tuple, list)):
                hidden = inputs[0]  # (B, T, H)
            else:
                hidden = inputs
            
            # 仅保存last token的hidden state
            # Only save last token's hidden state
            self.hidden_states[layer_id] = hidden[:, -1, :].detach()  # (B, H)
        return hook

    def register(self, model):
        """注册hooks Register hooks"""
        for i, layer in enumerate(model.model.layers):
            if i in self.target_layers:
                handle = layer.mlp.register_forward_hook(self._make_hook(i))
                self.handles.append(handle)

    def clear(self):
        """清空缓存 Clear cache"""
        self.hidden_states.clear()

    def remove(self):
        """移除hooks Remove hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

class AttentionHeadMasker:
    """
    Attention Head屏蔽器 - 在o_proj之前精确屏蔽
    Mask specific attention heads - precise masking before o_proj
    """
    def __init__(self):
        self.masked_layer = None
        self.masked_head = None
        self.handle = None

    def _make_pre_hook(self, layer_id, head_id, head_dim):
        """
        创建o_proj pre_hook，在线性变换前将指定head的通道置零
        Create o_proj pre_hook to zero out specified head channels before linear transform
        """
        def pre_hook(module, args):
            # args[0] is o_proj input: (bsz, seq_len, hidden_size)
            # At this point, each head's output is still in separate channel segments
            x = args[0]
            
            # 计算此head的通道范围
            # Calculate channel range for this head
            start_idx = head_id * head_dim
            end_idx = (head_id + 1) * head_dim
            
            # 将指定head的通道段置零
            # Zero out specified head's channel segment
            x[..., start_idx:end_idx] = 0.0
            
            # Return None to use modified original args
            return None
        return pre_hook

    def mask_head(self, model, layer_id, head_id):
        """
        屏蔽指定层的指定head - 在o_proj之前精确屏蔽
        Mask specified head in specified layer - precise masking before o_proj
        """
        self.unmask()  # 先清除之前的屏蔽 Clear previous mask first
        
        layer = model.model.layers[layer_id]
        num_heads = layer.self_attn.num_heads
        head_dim = layer.self_attn.head_dim
        
        if head_id >= num_heads:
            raise ValueError(f"Head {head_id} >= num_heads {num_heads}")
        
        self.masked_layer = layer_id
        self.masked_head = head_id
        
        # 在o_proj之前注册pre_hook以在线性变换前屏蔽指定head
        # Register pre_hook before o_proj to mask specified head before linear transform
        self.handle = layer.self_attn.o_proj.register_forward_pre_hook(
            self._make_pre_hook(layer_id, head_id, head_dim)
        )

    def unmask(self):
        """移除屏蔽 Remove mask"""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.masked_layer = None
        self.masked_head = None

# 核心分析函数 Core Analysis Functions
def compute_emotion_projection(hidden_states, emotion_vectors, target_layers):
    """
    计算情绪投影分数
    Compute emotion projection scores
    """
    projections = {}
    for layer_id in target_layers:
        if layer_id in hidden_states and layer_id < len(emotion_vectors):
            hidden = hidden_states[layer_id]  # (B, H)
            v_emotion = torch.tensor(emotion_vectors[layer_id], 
                                   device=hidden.device, dtype=hidden.dtype)  # (H,)
            
            # 计算点积：hidden · v_emotion (v_emotion已归一化)
            # Compute dot product: hidden · v_emotion (v_emotion already normalized)
            proj = torch.sum(hidden * v_emotion.unsqueeze(0), dim=1)  # (B,)
            projections[layer_id] = float(proj.mean())  # Take batch average
    return projections

def analyze_head_importance(model, tok, emotion, samples, emotion_dirs, target_layers, device):
    """
    分析attention head重要性
    Analyze attention head importance
    """
    collector = HiddenStateCollector(target_layers)
    masker = AttentionHeadMasker()
    
    collector.register(model)
    
    # 获取每层的head数量
    # Get head count for each layer
    layer_head_counts = {}
    for layer_id in target_layers:
        layer_head_counts[layer_id] = model.model.layers[layer_id].self_attn.num_heads
    
    results = defaultdict(lambda: defaultdict(list))  # {layer: {head: [importance_scores]}}
    baseline_projections = defaultdict(list)  # {layer: [baseline_scores]}
    
    print(f"[{emotion}] Analyzing {len(samples)} samples...")
    
    for sample in tqdm(samples, desc=f"Processing {emotion} samples"):
        scenario = sample.get("scenario", "")
        event = sample.get("event", "")
        
        prompt = apply_chat_template(tok, emotion, scenario, event)
        inputs = tok(prompt, return_tensors="pt").to(device)
        
        # 1. 计算基线投影分数
        # Compute baseline projection scores
        collector.clear()
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        
        baseline_proj = compute_emotion_projection(
            collector.hidden_states, emotion_dirs[emotion], target_layers
        )
        for layer_id, score in baseline_proj.items():
            baseline_projections[layer_id].append(score)
        
        # 2. 屏蔽每个head并计算投影变化
        # Mask each head and compute projection changes
        for layer_id in target_layers:
            num_heads = layer_head_counts[layer_id]
            baseline_score = baseline_proj.get(layer_id, 0.0)
            
            for head_id in range(num_heads):
                # 屏蔽此head Mask this head
                masker.mask_head(model, layer_id, head_id)
                
                # 重新前向传播 Re-run forward pass
                collector.clear()
                with torch.no_grad():
                    _ = model(**inputs, use_cache=False)
                
                # 计算屏蔽后的投影分数
                # Compute masked projection scores
                masked_proj = compute_emotion_projection(
                    collector.hidden_states, emotion_dirs[emotion], target_layers
                )
                masked_score = masked_proj.get(layer_id, 0.0)
                
                # 计算重要性：baseline - masked (下降越大 = head越重要)
                # Compute importance: baseline - masked (larger drop = more important head)
                importance = baseline_score - masked_score
                results[layer_id][head_id].append(importance)
                
                # 移除屏蔽 Remove mask
                masker.unmask()
    
    collector.remove()
    masker.unmask()
    
    return results, baseline_projections, layer_head_counts

def save_results(results, baseline_projections, layer_head_counts, emotion, out_dir):
    """
    保存分析结果
    Save analysis results
    """
    # 构建CSV数据
    # Build CSV data
    rows = []
    for layer_id in sorted(results.keys()):
        for head_id in range(layer_head_counts[layer_id]):
            if head_id in results[layer_id]:
                scores = results[layer_id][head_id]
                mean_importance = np.mean(scores)
                std_importance = np.std(scores)
                rows.append({
                    "emotion": emotion,
                    "layer": layer_id,
                    "head": head_id,
                    "mean_importance": mean_importance,
                    "std_importance": std_importance,
                    "samples": len(scores),
                    "baseline_mean": np.mean(baseline_projections[layer_id]) if layer_id in baseline_projections else 0.0
                })
    
    # 保存CSV
    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"head_importance_{emotion}.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path

# 主函数 Main
def main():
    ap = argparse.ArgumentParser(description="注意力头重要性分析 Attention Head Importance Analysis")
    ap.add_argument("--model_name", type=str, default="llama32_3b",
                   help="模型文件夹名称 Model folder name")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                   help="模型名称 Model name")
    ap.add_argument("--emotions", default="anger,sadness,happiness,fear,surprise,disgust",
                   help="情绪列表（逗号分隔）Emotion list (comma-separated)")
    ap.add_argument("--layers", default="", 
                   help="目标层，如 '10-20' 或 '10,15,20'；空=全层 Target layers; empty=all layers")
    ap.add_argument("--max_samples", type=int, default=1000, 
                   help="每个情绪分析的最大样本数 Maximum samples to analyze per emotion")
    ap.add_argument("--device", default="auto",
                   help="设备 Device")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","bfloat16","float16"],
                   help="数据类型 Data type")
    ap.add_argument("--seed", type=int, default=42,
                   help="随机种子 Random seed")
    args = ap.parse_args()
    
    seed_all(args.seed)
    
    # 确定输出目录
    # Determine output directory
    output_dir = ensure_dir(os.path.join("outputs", args.model_name, 
                                         "04_local_components_identification", 
                                         "attention_heads"))
    
    print(f"[+] Results will be saved to: {output_dir}")
    
    # 加载情绪方向向量
    # Load emotion direction vectors
    emotion_dirs, n_layers, H_dim, emotion_list = load_emo_dirs(args.model_name)
    print(f"[+] Loaded emotion directions: {n_layers} layers, {H_dim} hidden dim")
    
    # 确定目标层
    # Determine target layers
    if args.layers.strip():
        if "-" in args.layers:
            start, end = map(int, args.layers.split("-"))
            target_layers = list(range(start, end + 1))
        else:
            target_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        target_layers = list(range(n_layers))
    
    print(f"[+] Target layers: {target_layers}")
    
    # 设置数据类型
    # Set data type
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    torch_dtype = dtype_map[args.dtype]
    
    # 加载模型
    # Load model
    device = "cuda:0" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=HF_TOKEN if HF_TOKEN else True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch_dtype, 
        device_map={"": device},
        token=HF_TOKEN if HF_TOKEN else True
    )
    model.eval()
    print(f"[+] Model loaded on {device}")
    
    # 处理每个情绪
    # Process emotions
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    
    for emotion in emotions:
        print(f"\n{'='*60}")
        print(f"ANALYZING EMOTION: {emotion.upper()}")
        print(f"{'='*60}")
        
        # 加载样本（all valence）
        # Load samples (all valence)
        samples = load_accepted_samples(args.model_name, emotion, args.max_samples)
        
        if not samples:
            print(f"[!] No samples found for {emotion}")
            continue
        
        # 样本信息报告
        # Sample info report
        print(f"[+] Loaded {len(samples)} samples for {emotion.upper()}")
        print(f"    - Target layers: {len(target_layers)} layers ({min(target_layers)}-{max(target_layers)})")
        
        # 计算总测试数
        # Calculate total test count
        total_heads = sum(model.model.layers[i].self_attn.num_heads for i in target_layers)
        total_forward_passes = len(samples) * (1 + total_heads)  # baseline + heads_masked
        print(f"    - Total attention heads to test: {total_heads}")
        print(f"    - Total forward passes needed: {total_forward_passes:,}")
        print(f"    - Estimated time: {total_forward_passes * 0.1:.1f}s (approx.)")
        
        # 分析head重要性
        # Analyze head importance
        print(f"[+] Starting analysis...")
        results, baselines, head_counts = analyze_head_importance(
            model, tok, emotion, samples, emotion_dirs, target_layers, device
        )
        
        # 保存结果
        # Save results
        csv_path = save_results(results, baselines, head_counts, emotion, output_dir)
        
        print(f"[+] Results saved for {emotion}")
        print(f"    CSV: {csv_path}")
    
    print(f"\n[✓] All done! Results saved to:")
    print(f"    {output_dir}")

if __name__ == "__main__":
    main()
