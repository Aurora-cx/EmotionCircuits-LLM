# -*- coding: utf-8 -*-
# scripts/05_emotion_diff_vector_computation/2_compute_emotion_mlp_diff.py
"""
计算情绪差分向量（组内差分）
Compute Emotion Difference Vectors (Within-Sample Contrast)

组内差分（within-sample contrast）再平均方法：
Within-sample contrast then average method:
对每个组g、每层L：
For each group g, each layer L:
  1) 取该组所有情绪的向量并按同情绪内部平均（若同情绪有多个文件）
     Take all emotion vectors in the group and average within the same emotion (if multiple files)
  2) 计算组内全情绪均值 m_g[L]
     Calculate the within-group all-emotion mean m_g[L]
  3) 对每个情绪e：delta_g,e[L] = h_g,e[L] - m_g[L]
     For each emotion e: delta_g,e[L] = h_g,e[L] - m_g[L]
最后：对所有组的 delta_g,e[L] 求平均，得到 emo_diff[e][L]
Finally: Average all groups' delta_g,e[L] to get emo_diff[e][L]

- 输入 Input: outputs/{model_name}/05_emotion_diff_vector_computation/down_proj_input/*.pt
- 输出 Output: outputs/{model_name}/05_emotion_diff_vector_computation/mlp_emotion_diff/
"""

import os, re, json, argparse
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict

# ============== 路径与基本配置 / Paths and Basic Configuration ==============
# 工作目录：项目根目录
# Working directory: project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL_NAME = "llama32_3b"
# 六个基础情绪
# Six basic emotions
EMOTIONS = ["anger","sadness","happiness","fear","surprise","disgust"]
# 三个效价
# Three valences
VALENCES = ["positive","neutral","negative"]
N_LAYERS = 28  # Llama-3.2-3B 层数 / Number of layers
D_FF = 8192    # MLP中间层维度 / MLP intermediate dimension

# 解析文件名：skeleton_id__valence
# Parse filename: skeleton_id__valence
FNAME_RE = re.compile(
    r"^(?P<skeleton>.+?)\_\_(?P<valence>positive|neutral|negative)\.pt$"
)

def ensure_dir(p):
    """
    确保目录存在
    Ensure directory exists
    """
    os.makedirs(p, exist_ok=True)
    return p

def parse_fname(fn: str):
    """
    解析文件名，提取skeleton_id和valence
    Parse filename to extract skeleton_id and valence
    """
    m = FNAME_RE.match(fn)
    if not m:
        return None
    return m.group("skeleton"), m.group("valence")

def load_pt_file(pt_path):
    """
    加载.pt文件，返回所有6个情绪的激活值
    Load .pt file and return activations for all 6 emotions
    
    返回格式 Return format:
      {emotion: tensor(n_layers, d_ff) for emotion in EMOTIONS}
    """
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    # 从 "hidden_last_all_layers" 字段中提取每个情绪的激活值
    # Extract activations for each emotion from "hidden_last_all_layers" field
    hidden_dict = data.get("hidden_last_all_layers", {})
    
    result = {}
    for emotion in EMOTIONS:
        if emotion in hidden_dict:
            # hidden_dict[emotion] 形状: (n_layers, d_ff)
            # hidden_dict[emotion] shape: (n_layers, d_ff)
            result[emotion] = hidden_dict[emotion].float().cpu()
    
    return result

def build_groups(input_dir):
    """
    构建分组字典
    Build groups dictionary
    
    返回 Return:
      groups: dict[skeleton_id] -> dict[emotion] -> tensor(n_layers, d_ff)
      skeleton_id是全局唯一的，一个skeleton_id对应一组6个情绪
      skeleton_id is globally unique, one skeleton_id corresponds to one group of 6 emotions
    """
    groups = defaultdict(dict)
    
    for fn in os.listdir(input_dir):
        if not fn.endswith(".pt"): 
            continue
        
        parsed = parse_fname(fn)
        if parsed is None:
            continue
        
        skeleton_id, valence = parsed
        pt_path = os.path.join(input_dir, fn)
        
        # 加载该文件中的所有情绪激活值
        # Load all emotion activations from this file
        emotion_tensors = load_pt_file(pt_path)
        
        # 将每个情绪的激活值存入对应的skeleton组中
        # Store each emotion's activations in the corresponding skeleton group
        for emotion, tensor in emotion_tensors.items():
            groups[skeleton_id][emotion] = tensor
    
    return groups

def compute_within_contrast(input_dir, output_dir, emotions, layers, winsor=None):
    """
    组内差分再平均
    Within-sample contrast then average
    
    对每个组g、每层L：
    For each group g, each layer L:
      1) 取该组所有情绪的向量并按同情绪内部平均（若同情绪有多个文件）
         Take all emotion vectors in the group and average within the same emotion (if multiple files)
      2) 计算组内全情绪均值 m_g[L]
         Calculate the within-group all-emotion mean m_g[L]
      3) 对每个情绪e：delta_g,e[L] = h_g,e[L] - m_g[L]
         For each emotion e: delta_g,e[L] = h_g,e[L] - m_g[L]
    最后：对所有组的 delta_g,e[L] 求平均，得到 emo_diff[e][L]
    Finally: Average all groups' delta_g,e[L] to get emo_diff[e][L]
    
    同时，为了兼容/对照，保留：
    Also, for compatibility/comparison, retain:
      - emo_mean[e][L]：所有组内该情绪的向量再平均（不是差分）
        emo_mean[e][L]: Average of all groups' vectors for that emotion (not diff)
      - all_mean[L]：所有组的组均值 m_g[L] 再平均（等价于全体均值）
        all_mean[L]: Average of all groups' group means m_g[L] (equivalent to overall mean)
    """
    groups = build_groups(input_dir)

    # 累积器
    # Accumulators
    sum_diff = {e: {L: np.zeros(D_FF, np.float64) for L in layers} for e in emotions}
    cnt_diff = {e: {L: 0 for L in layers} for e in emotions}

    sum_emo = {e: {L: np.zeros(D_FF, np.float64) for L in layers} for e in emotions}
    cnt_emo = {e: {L: 0 for L in layers} for e in emotions}

    sum_allmean = {L: np.zeros(D_FF, np.float64) for L in layers}
    cnt_allmean = {L: 0 for L in layers}

    # 遍历每个组
    # Iterate through each group
    for skeleton_id, emotion_tensors in groups.items():
        # 该组内每个情绪、每层的向量
        # Vectors for each emotion and each layer in this group
        # h_e[emotion][layer] : np.ndarray(d_ff,)
        h_e = {e: {} for e in emotions}
        has_any = False

        for e in emotions:
            if e not in emotion_tensors:
                continue
            has_any = True
            
            # emotion_tensors[e] 形状: (n_layers, d_ff)
            # emotion_tensors[e] shape: (n_layers, d_ff)
            tensor = emotion_tensors[e]
            
            for L in layers:
                if L >= tensor.shape[0]:
                    continue
                
                # 该组-该情绪-该层的向量
                # Vector for this group-emotion-layer
                vec = tensor[L].numpy().astype(np.float32)  # (d_ff,)
                h_e[e][L] = vec
                
                # 同时累计全局的 emo_mean
                # Accumulate global emo_mean
                sum_emo[e][L] += vec
                cnt_emo[e][L] += 1

        if not has_any:
            continue

        # 计算本组的组内全情绪均值（每层）
        # Calculate within-group all-emotion mean for each layer
        for L in layers:
            # 收集本层在该组里实际存在的情绪向量
            # Collect emotion vectors that actually exist in this group for this layer
            layer_vecs = [h_e[e][L] for e in emotions if L in h_e[e]]
            if len(layer_vecs) == 0:
                continue
            
            # 组均值
            # Group mean
            m_g_L = np.mean(layer_vecs, axis=0).astype(np.float32)
            sum_allmean[L] += m_g_L
            cnt_allmean[L] += 1

            # 组内差分并累计
            # Within-group contrast and accumulate
            for e in emotions:
                if L not in h_e[e]:
                    continue
                delta = (h_e[e][L] - m_g_L).astype(np.float32)
                sum_diff[e][L] += delta
                cnt_diff[e][L] += 1

    # 计算最终结果并保存
    # Calculate final results and save
    out_base = ensure_dir(output_dir)
    # 主结果：组内差分再平均
    # Main result: within-sample contrast then average
    out_diff = ensure_dir(os.path.join(out_base, "emo_diff"))

    pack = {}
    summary = {}

    for e in emotions:
        summary[e] = {"layers": {}}
        dir_e_diff = ensure_dir(os.path.join(out_diff, e))

        for L in layers:
            # 组内差分再平均（主结果）
            # Within-sample contrast then average (main result)
            if cnt_diff[e][L] > 0:
                diff = (sum_diff[e][L] / cnt_diff[e][L]).astype(np.float32)
            else:
                diff = np.zeros(D_FF, np.float32)

            # 可选winsor（只对diff裁剪）
            # Optional winsor (only clip diff)
            if winsor is not None and winsor > 0.0:
                lo, hi = np.percentile(diff, [winsor, 100 - winsor])
                diff = np.clip(diff, lo, hi)

            # 保存逐层.npy文件
            # Save layer-by-layer .npy files
            np.save(os.path.join(dir_e_diff, f"layer{L}.npy"), diff)

            # 打包（只保存diff）
            # Package (only save diff)
            pack[f"{e}/diff/L{L}"] = diff

            # 记录summary
            # Record summary
            summary[e]["layers"][str(L)] = dict(
                # 该情绪在多少组出现
                # Number of groups with this emotion
                n_groups_with_e=int(cnt_emo[e][L]),
                # 参与all_mean的组数
                # Number of groups participating in all_mean
                n_groups_total=int(cnt_allmean[L]),
                # 该情绪组内差分的样本数
                # Number of samples for this emotion's within-group contrast
                n_groups_for_diff=int(cnt_diff[e][L]),
                d_ff=int(D_FF)
            )

        # 保存每个情绪的summary
        # Save summary for each emotion
        with open(os.path.join(out_base, f"emo_diff_summary_{e}.json"), "w", encoding="utf-8") as f:
            json.dump(summary[e], f, ensure_ascii=False, indent=2)

    # 保存总打包
    # Save overall package
    np.savez_compressed(os.path.join(out_base, "emo_diff_all.npz"), **pack)

    # 总summary
    # Overall summary
    with open(os.path.join(out_base, "emo_diff_summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[✓] Saved within-sample diffs to {out_base}")
    return out_base

def main():
    """
    主函数
    Main function
    """
    ap = argparse.ArgumentParser(
        description="计算情绪差分向量（组内差分）/ Compute emotion difference vectors (within-sample contrast)"
    )
    ap.add_argument(
        "--model_name", 
        default=MODEL_NAME,
        help="模型名称 / Model name"
    )
    ap.add_argument(
        "--input_dir", 
        default=None,
        help="输入目录（包含.pt文件），默认为outputs/{model_name}/05_emotion_diff_vector_computation/down_proj_input / Input directory (containing .pt files)"
    )
    ap.add_argument(
        "--output_dir", 
        default=None,
        help="输出目录，默认为outputs/{model_name}/05_emotion_diff_vector_computation/mlp_emotion_diff / Output directory"
    )
    ap.add_argument(
        "--emotions", 
        default="anger,sadness,happiness,fear,surprise,disgust",
        help="要计算的情绪列表（逗号分隔）/ List of emotions to compute (comma-separated)"
    )
    ap.add_argument(
        "--layers", 
        default="", 
        help="要计算的层列表（逗号分隔），空表示所有层0..27 / List of layers to compute (comma-separated), empty means all layers 0..27"
    )
    ap.add_argument(
        "--winsor", 
        type=float, 
        default=0.0, 
        help="winsorize百分位（如1.0表示裁剪到[1%%,99%%]）/ Winsorize percentile (e.g., 1.0 means clip to [1%%,99%%])"
    )
    args = ap.parse_args()

    # 设置默认路径
    # Set default paths
    if args.input_dir is None:
        args.input_dir = PROJECT_ROOT / f"outputs/{args.model_name}/05_emotion_diff_vector_computation/down_proj_input"
    else:
        args.input_dir = Path(args.input_dir)
    
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / f"outputs/{args.model_name}/05_emotion_diff_vector_computation/mlp_emotion_diff"
    else:
        args.output_dir = Path(args.output_dir)

    # 解析层列表
    # Parse layer list
    if args.layers.strip():
        layers = [int(x) for x in args.layers.split(",") if x.strip()!=""]
    else:
        layers = list(range(N_LAYERS))

    # 解析情绪列表
    # Parse emotion list
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Emotions: {emotions}")
    print(f"Layers: {len(layers)} layers (0..{max(layers)})")
    print(f"Winsor percentile: {args.winsor}")
    print()

    # 计算差分向量
    # Compute difference vectors
    out_dir = compute_within_contrast(
        str(args.input_dir), 
        str(args.output_dir), 
        emotions, 
        layers, 
        winsor=args.winsor
    )
    print(f"[Done] Within-sample emotion diffs written under: {out_dir}")

if __name__ == "__main__":
    main()