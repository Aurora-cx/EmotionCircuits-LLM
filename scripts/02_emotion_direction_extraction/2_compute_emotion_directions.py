# scripts/02_emotion_direction_extraction/2_compute_emotion_directions.py
# -*- coding: utf-8 -*-
"""
计算情绪方向向量
Compute Emotion Direction Vectors

从 residual_dump/attention/ 和 mlp/ 里分别计算情绪方向向量
Compute emotion direction vectors from residual_dump/attention/ and mlp/

方法 Method：
- 每个组内做 6情绪 - 组均值 → 去事件化
  Within each group: 6 emotions - group mean → de-contextualization
- 跨组取均值得到情绪质心
  Cross-group mean to get emotion centroids
- 去全局均值后归一化，得到情绪方向向量 v_e^{(l)}
  Remove global mean and normalize to get emotion direction vectors v_e^{(l)}
- 位置都是残差流的不同位置插桩
  Positions are different points in the residual stream

输入 Input: outputs/{model_name}/02_emotion_directions/residual_dump/attention/ 和 mlp/
输出 Output: outputs/{model_name}/02_emotion_directions/emo_directions_mlp.pt 和 emo_directions_attention.pt
"""

import os, argparse
from pathlib import Path
import numpy as np
import torch

# 6种情绪
# 6 emotions
EMOS6 = ["anger","sadness","happiness","fear","surprise","disgust"]

def load_groups(data_dir: Path, data_type: str):
    """
    加载指定类型的数据，返回 groups: list[dict emo->[L,H]]
    Load data of specified type, return groups: list[dict emo->[L,H]]
    """
    pts = sorted(data_dir.glob("*.pt"))
    
    print(f"Loading {len(pts)} files for {data_type} emotion direction calculation...")
    
    # 按skeleton_id和valence分组，收集所有情绪数据
    # Group by skeleton_id and valence, collect all emotion data
    from collections import defaultdict
    grouped_data = defaultdict(lambda: defaultdict(dict))
    
    for p in pts:
        try:
            rec = torch.load(p, map_location="cpu")
        except Exception as e:
            print(f"[LOAD-ERROR] {p.name}: {e}")
            continue
            
        H = rec.get("hidden_last_all_layers", {})
        if not H:
            print(f"[SKIP] {p.name}: no hidden_last_all_layers")
            continue
            
        # 从文件名提取skeleton_id和valence
        # Extract skeleton_id and valence from filename
        # 格式 Format: skeleton_id__valence.pt
        filename = p.stem
        if '__' not in filename:
            print(f"[SKIP] {p.name}: invalid filename format")
            continue
            
        skeleton_id, valence = filename.rsplit('__', 1)
        group_key = f"{skeleton_id}__{valence}"
        
        # 从hidden_last_all_layers中获取情绪数据
        # Get emotion data from hidden_last_all_layers
        for emotion, tensor in H.items():
            if emotion in EMOS6:
                grouped_data[group_key][emotion] = tensor.numpy().astype(np.float32)
    
    # 转换为groups格式，只保留包含所有6个情绪的组
    # Convert to groups format, keep only groups with all 6 emotions
    groups = []
    for group_key, emotions in grouped_data.items():
        if all(e in emotions for e in EMOS6):
            groups.append({e: emotions[e] for e in EMOS6})
        else:
            missing = [e for e in EMOS6 if e not in emotions]
            print(f"[SKIP] {group_key}: missing emotions {missing}")
    
    if not groups:
        raise RuntimeError(f"No usable groups found in {data_dir}")
    
    L = next(iter(groups))[EMOS6[0]].shape[0]
    Hdim = next(iter(groups))[EMOS6[0]].shape[1]
    
    print(f"Loaded {len(groups)} groups for {data_type} | layers={L}, H={Hdim}")
    return groups, L, Hdim

def build_emotion_directions(groups, L, Hdim):
    """
    计算情绪方向向量
    Compute emotion direction vectors
    返回 Return: dirs: dict emo->[L,H]
    """
    # 组内去均值
    # Remove within-group mean
    centered = []
    for g in groups:
        mu = np.stack([g[e] for e in EMOS6], 0).mean(axis=0)  # [L,H]
        centered.append({e: (g[e] - mu).astype(np.float32) for e in EMOS6})

    # 跨组情绪质心
    # Cross-group emotion centroids
    centroids = {e: np.stack([cg[e] for cg in centered], 0).mean(axis=0) for e in EMOS6}  # emo->[L,H]

    # 去全局均值
    # Remove global mean
    all_mean = np.stack(list(centroids.values()), 0).mean(axis=0)  # [L,H]

    # 归一化得到情绪方向向量
    # Normalize to get emotion direction vectors
    dirs = {}
    for e in EMOS6:
        v = centroids[e] - all_mean
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12  # [L,1]
        dirs[e] = (v / n).astype(np.float32)  # [L,H]
    return dirs

def compute_mlp_emotion_directions(mlp_dir, out_dir):
    """
    计算MLP的情绪方向向量
    Compute MLP emotion direction vectors
    """
    print("\n=== Computing MLP Emotion Directions ===")
    
    try:
        groups, L, Hdim = load_groups(mlp_dir, "MLP")
        dirs = build_emotion_directions(groups, L, Hdim)
        
        # 保存MLP情绪向量
        # Save MLP emotion directions
        out_path = out_dir / "emo_directions_mlp.pt"
        torch.save({
            "dirs": dirs, 
            "layers": L, 
            "hidden": Hdim, 
            "emotions": EMOS6,
            "type": "mlp"
        }, out_path)
        print(f"[✓] Saved MLP emotion directions to {out_path}")
        
        # 打印检查
        # Print verification
        print("MLP Emotion Directions:")
        for e in EMOS6:
            arr = dirs[e]
            print(f"  {e:>9s} | shape={arr.shape} | mean-norm={np.mean(np.linalg.norm(arr, axis=1)):.4f}")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to compute MLP emotion directions: {e}")
        return False

def compute_attention_emotion_directions(attention_dir, out_dir):
    """
    计算Attention的情绪方向向量
    Compute Attention emotion direction vectors
    """
    print("\n=== Computing Attention Emotion Directions ===")
    
    try:
        groups, L, Hdim = load_groups(attention_dir, "Attention")
        dirs = build_emotion_directions(groups, L, Hdim)
        
        # 保存Attention情绪向量
        # Save Attention emotion directions
        out_path = out_dir / "emo_directions_attention.pt"
        torch.save({
            "dirs": dirs, 
            "layers": L, 
            "hidden": Hdim, 
            "emotions": EMOS6,
            "type": "attention"
        }, out_path)
        print(f"[✓] Saved Attention emotion directions to {out_path}")
        
        # 打印检查
        # Print verification
        print("Attention Emotion Directions:")
        for e in EMOS6:
            arr = dirs[e]
            print(f"  {e:>9s} | shape={arr.shape} | mean-norm={np.mean(np.linalg.norm(arr, axis=1)):.4f}")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to compute Attention emotion directions: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama32_3b",
                       help="模型文件夹名称 Model folder name")
    args = parser.parse_args()
    
    print("计算MLP和Attention的情绪方向向量")
    print("Computing emotion directions for MLP and Attention")
    print("=" * 60)
    
    # 构建路径
    # Build paths
    model_name = args.model_name
    attention_dir = Path("outputs") / model_name / "02_emotion_directions" / "residual_dump" / "attention"
    mlp_dir = Path("outputs") / model_name / "02_emotion_directions" / "residual_dump" / "mlp"
    out_dir = Path("outputs") / model_name / "02_emotion_directions"
    
    # 检查输入目录是否存在
    # Check if input directories exist
    if not attention_dir.exists():
        print(f"[ERROR] Attention directory not found: {attention_dir}")
        return
    
    if not mlp_dir.exists():
        print(f"[ERROR] MLP directory not found: {mlp_dir}")
        return
    
    # 计算MLP情绪方向向量
    # Compute MLP emotion directions
    mlp_success = compute_mlp_emotion_directions(mlp_dir, out_dir)
    
    # 计算Attention情绪方向向量
    # Compute Attention emotion directions
    attention_success = compute_attention_emotion_directions(attention_dir, out_dir)
    
    # 总结
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"MLP emotion directions: {'✓ SUCCESS' if mlp_success else '✗ FAILED'}")
    print(f"Attention emotion directions: {'✓ SUCCESS' if attention_success else '✗ FAILED'}")
    
    if mlp_success and attention_success:
        print(f"\n情绪方向向量已保存到 All emotion directions saved to: {out_dir}")
        print("- emo_directions_mlp.pt: MLP情绪方向向量 MLP-based emotion directions")
        print("- emo_directions_attention.pt: Attention情绪方向向量 Attention-based emotion directions")
    else:
        print("\n部分计算失败，请检查上面的错误信息")
        print("Some computations failed. Check the error messages above.")

if __name__ == "__main__":
    main()
