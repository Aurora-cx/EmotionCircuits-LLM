# -*- coding: utf-8 -*-
# scripts/05_emotion_diff_vector_computation/3_compute_emotion_attn_diff.py
"""
计算Attention情绪差分向量（组内差分）
Compute Attention Emotion Difference Vectors (Within-Sample Contrast)

基于o_proj输入激活值计算情绪差分向量，使用组内差分再平均方法
Compute emotion difference vectors based on o_proj input activations using within-sample contrast then average method

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

- 输入 Input: outputs/{model_name}/05_emotion_diff_vector_computation/o_proj_input/*.pt
- 输出 Output: outputs/{model_name}/05_emotion_diff_vector_computation/attention_emotion_diff/
"""

import os, json, argparse
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
EMOS6 = ["anger","sadness","happiness","fear","surprise","disgust"]
LAYERS = list(range(28))  # L0-L27, Llama-3.2-3B层数 / Number of layers
D_HIDDEN = 3072  # Llama-3.2-3B的hidden_size / Hidden size

# ============== 工具函数 / Utility Functions ==============
def ensure_dir(path):
    """
    确保目录存在
    Ensure directory exists
    """
    os.makedirs(path, exist_ok=True)
    return path

def load_pt_file(pt_path):
    """
    加载.pt文件，返回所有6个情绪的激活值
    Load .pt file and return activations for all 6 emotions
    
    返回格式 Return format:
      {emotion: tensor(n_layers, d_hidden) for emotion in EMOS6}
    """
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    # 从 "hidden_last_all_layers" 字段中提取每个情绪的激活值
    # Extract activations for each emotion from "hidden_last_all_layers" field
    hidden_dict = data.get("hidden_last_all_layers", {})
    
    result = {}
    for emotion in EMOS6:
        if emotion in hidden_dict:
            # hidden_dict[emotion] 形状: (n_layers, d_hidden)
            # hidden_dict[emotion] shape: (n_layers, d_hidden)
            result[emotion] = hidden_dict[emotion].float().cpu()
    
    return result

def build_groups(data_dir):
    """
    构建分组数据结构
    Build groups data structure
    
    返回 Return:
      groups: dict[skeleton_id] -> dict[emotion] -> tensor(n_layers, d_hidden)
      skeleton_id是全局唯一的，一个skeleton_id对应一组6个情绪
      skeleton_id is globally unique, one skeleton_id corresponds to one group of 6 emotions
    """
    groups = defaultdict(dict)
    
    for pt_file in data_dir.glob("*.pt"):
        filename = pt_file.stem
        # 解析文件名: skeleton_id__valence
        # Parse filename: skeleton_id__valence
        if "__" not in filename:
            continue
        
        skeleton_id = filename.split("__")[0]
        
        # 加载该文件中的所有情绪激活值
        # Load all emotion activations from this file
        emotion_tensors = load_pt_file(pt_file)
        
        # 将每个情绪的激活值存入对应的skeleton组中
        # Store each emotion's activations in the corresponding skeleton group
        for emotion, tensor in emotion_tensors.items():
            groups[skeleton_id][emotion] = tensor
    
    return groups

def compute_within_contrast(input_dir, output_dir, emotions, layers, normalize=True):
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
    """
    groups = build_groups(Path(input_dir))
    
    # 累积器
    # Accumulators
    sum_diff = {e: {L: np.zeros(D_HIDDEN, np.float64) for L in layers} for e in emotions}
    cnt_diff = {e: {L: 0 for L in layers} for e in emotions}
    
    sum_emo = {e: {L: np.zeros(D_HIDDEN, np.float64) for L in layers} for e in emotions}
    cnt_emo = {e: {L: 0 for L in layers} for e in emotions}
    
    sum_allmean = {L: np.zeros(D_HIDDEN, np.float64) for L in layers}
    cnt_allmean = {L: 0 for L in layers}
    
    print(f"Processing {len(groups)} groups...")
    
    # 遍历每个组
    # Iterate through each group
    for skeleton_id, emotion_tensors in groups.items():
        # 该组内每个情绪、每层的向量
        # Vectors for each emotion and each layer in this group
        # h_e[emotion][layer] : np.ndarray(d_hidden,)
        h_e = {e: {} for e in emotions}
        has_any = False
        
        for e in emotions:
            if e not in emotion_tensors:
                continue
            has_any = True
            
            # emotion_tensors[e] 形状: (n_layers, d_hidden)
            # emotion_tensors[e] shape: (n_layers, d_hidden)
            tensor = emotion_tensors[e]
            
            for L in layers:
                if L >= tensor.shape[0]:
                    continue
                
                # 该组-该情绪-该层的向量
                # Vector for this group-emotion-layer
                layer_vec = tensor[L].numpy().astype(np.float32)  # (d_hidden,)
                h_e[e][L] = layer_vec
                
                # 同时累计全局的 emo_mean
                # Accumulate global emo_mean
                sum_emo[e][L] += layer_vec
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
    out_diff = ensure_dir(os.path.join(out_base, "emo_diff"))
    
    pack = {}
    summary = {}
    
    for e in emotions:
        summary[e] = {"layers": {}}
        dir_e_diff = ensure_dir(os.path.join(out_diff, e))
        
        for L in layers:
            # 计算差分向量
            # Calculate difference vector
            if cnt_diff[e][L] > 0:
                emo_diff_L = (sum_diff[e][L] / cnt_diff[e][L]).astype(np.float32)
            else:
                emo_diff_L = np.zeros(D_HIDDEN, dtype=np.float32)
            
            # 可选：L2归一化
            # Optional: L2 normalization
            if normalize:
                norm = np.linalg.norm(emo_diff_L)
                if norm > 1e-8:  # 避免除零 / Avoid division by zero
                    emo_diff_L = emo_diff_L / norm
            
            # 保存差分向量
            # Save difference vector
            diff_file = os.path.join(dir_e_diff, f"layer{L}.npy")
            np.save(diff_file, emo_diff_L)
            
            # 保存到pack中（用于npz格式）
            # Save to pack (for npz format)
            pack[f"{e}/diff/L{L}"] = emo_diff_L
            
            # 统计信息
            # Statistics
            norm = np.linalg.norm(emo_diff_L)
            summary[e]["layers"][str(L)] = {
                # 该情绪组内差分的样本数
                # Number of samples for this emotion's within-group contrast
                "n_groups_for_diff": int(cnt_diff[e][L]),
                # 该情绪在多少组出现
                # Number of groups with this emotion
                "n_groups_with_e": int(cnt_emo[e][L]),
                # 参与all_mean的组数
                # Number of groups participating in all_mean
                "n_groups_total": int(cnt_allmean[L]),
                "norm": float(norm),
                "mean": float(np.mean(emo_diff_L)),
                "std": float(np.std(emo_diff_L)),
                "d_hidden": int(D_HIDDEN)
            }
        
        # 保存每个情绪的summary
        # Save summary for each emotion
        with open(os.path.join(out_base, f"emo_diff_summary_{e}.json"), "w", encoding="utf-8") as f:
            json.dump(summary[e], f, ensure_ascii=False, indent=2)
    
    # 保存总打包npz文件
    # Save overall package npz file
    npz_file = os.path.join(out_base, "emo_diff_all.npz")
    np.savez_compressed(npz_file, **pack)
    
    # 保存总统计信息
    # Save overall summary
    summary_file = os.path.join(out_base, "emo_diff_summary_all.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] Saved attention emotion diff vectors to: {out_base}")
    print(f"  NPZ file: {npz_file}")
    print(f"  Summary file: {summary_file}")
    
    return summary

def main():
    """
    主函数
    Main function
    """
    parser = argparse.ArgumentParser(
        description="计算Attention情绪差分向量（组内差分）/ Compute attention emotion difference vectors (within-sample contrast)"
    )
    parser.add_argument(
        "--model_name", 
        default=MODEL_NAME,
        help="模型名称 / Model name"
    )
    parser.add_argument(
        "--input_dir", 
        default=None,
        help="输入目录（包含.pt文件），默认为outputs/{model_name}/05_emotion_diff_vector_computation/o_proj_input / Input directory (containing .pt files)"
    )
    parser.add_argument(
        "--output_dir", 
        default=None,
        help="输出目录，默认为outputs/{model_name}/05_emotion_diff_vector_computation/attention_emotion_diff / Output directory"
    )
    parser.add_argument(
        "--emotions", 
        default="anger,sadness,happiness,fear,surprise,disgust",
        help="要计算的情绪列表（逗号分隔）/ List of emotions to compute (comma-separated)"
    )
    parser.add_argument(
        "--layers", 
        default="", 
        help="要计算的层列表（逗号分隔），空表示所有层0..27 / List of layers to compute (comma-separated), empty means all layers 0..27"
    )
    parser.add_argument(
        "--normalize", 
        action="store_true", 
        default=True, 
        help="是否L2归一化差分向量（默认：True）/ Whether to L2 normalize the difference vectors (default: True)"
    )
    parser.add_argument(
        "--no_normalize", 
        action="store_true", 
        help="禁用L2归一化 / Disable L2 normalization"
    )
    args = parser.parse_args()
    
    # 设置默认路径
    # Set default paths
    if args.input_dir is None:
        args.input_dir = PROJECT_ROOT / f"outputs/{args.model_name}/05_emotion_diff_vector_computation/o_proj_input"
    else:
        args.input_dir = Path(args.input_dir)
    
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / f"outputs/{args.model_name}/05_emotion_diff_vector_computation/attention_emotion_diff"
    else:
        args.output_dir = Path(args.output_dir)
    
    # 解析层列表
    # Parse layer list
    if args.layers.strip():
        layers = [int(x) for x in args.layers.split(",") if x.strip()!=""]
    else:
        layers = LAYERS
    
    # 解析情绪列表
    # Parse emotion list
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    
    # 处理归一化参数
    # Handle normalization parameter
    normalize = args.normalize and not args.no_normalize
    
    print("Building attention emotion directions from o_proj input activations...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Emotions: {emotions}")
    print(f"Layers: {len(layers)} layers (0..{max(layers)})")
    print(f"L2 normalization: {normalize}")
    print()
    
    # 检查输入目录
    # Check input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist!")
        print("Please run script 1 first to generate o_proj input activations.")
        return
    
    # 检查输入文件
    # Check input files
    pt_files = list(args.input_dir.glob("*.pt"))
    if not pt_files:
        print(f"Error: No .pt files found in {args.input_dir}")
        return
    
    print(f"Found {len(pt_files)} .pt files")
    
    # 计算情绪差分向量
    # Compute emotion difference vectors
    summary = compute_within_contrast(
        str(args.input_dir), 
        str(args.output_dir), 
        emotions, 
        layers, 
        normalize=normalize
    )
    
    # 打印统计信息
    # Print statistics
    print("\n=== Summary ===")
    for emotion in emotions:
        print(f"\n{emotion}:")
        # 显示部分层的统计
        # Show statistics for selected layers
        for layer in [0, 5, 10, 15, 20, 25, 27]:
            if str(layer) in summary[emotion]["layers"]:
                stats = summary[emotion]["layers"][str(layer)]
                print(f"  L{layer}: n_groups={stats['n_groups_for_diff']}, norm={stats['norm']:.4f}, mean={stats['mean']:.6f}, std={stats['std']:.4f}")
    
    print(f"\n[Done] Attention emotion diff vectors saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
