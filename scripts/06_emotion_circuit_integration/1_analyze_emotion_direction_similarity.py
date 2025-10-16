# -*- coding: utf-8 -*-
# scripts/06_emotion_circuit_integration/1_analyze_emotion_direction_similarity.py
"""
分析情绪方向向量在整个残差流中的跨层变化
Analyze cross-layer changes of emotion direction vectors in residual stream

计算56个位置（28层 × 2位置：Attention + MLP）的余弦相似度矩阵
Compute cosine similarity matrix for 56 positions (28 layers × 2 positions: Attention + MLP)

- 输入 Input: outputs/{model_name}/02_emotion_directions/emo_directions_*.pt
- 输出 Output: outputs/{model_name}/06_emotion_circuit_integration/emotion_direction_similarity/
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ============== 路径与基本配置 / Paths and Basic Configuration ==============
# 工作目录：项目根目录
# Working directory: project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

# 设置matplotlib支持中文
# Set matplotlib to support Chinese fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

MODEL_NAME = "llama32_3b"
EMOS6 = ["anger", "sadness", "happiness", "fear", "surprise", "disgust"]

# ============== 工具函数 / Utility Functions ==============
def ensure_dir(p):
    """
    确保目录存在
    Ensure directory exists
    """
    os.makedirs(p, exist_ok=True)
    return p

# ============== 数据加载器 / Data Loaders ==============
def load_emotion_directions(model_name=MODEL_NAME):
    """
    加载Attention和MLP的情绪方向向量
    Load Attention and MLP emotion direction vectors
    """
    print("Loading emotion direction vectors...")
    
    data_dir = PROJECT_ROOT / "outputs" / model_name / "02_emotion_directions"
    attn_path = data_dir / "emo_directions_attention.pt"
    mlp_path = data_dir / "emo_directions_mlp.pt"
    
    attn_data = torch.load(attn_path, map_location="cpu", weights_only=False)
    mlp_data = torch.load(mlp_path, map_location="cpu", weights_only=False)
    
    print(f"  Attention: {attn_data['layers']} layers, {attn_data['hidden']} dims")
    print(f"  MLP: {mlp_data['layers']} layers, {mlp_data['hidden']} dims")
    
    return attn_data['dirs'], mlp_data['dirs']

def build_residual_stream_sequence(attn_dirs, mlp_dirs):
    """
    构建残差流序列：Attn0-MLP0-Attn1-MLP1-...-Attn27-MLP27
    Build residual stream sequence: Attn0-MLP0-Attn1-MLP1-...-Attn27-MLP27
    
    返回：dict of {emotion: [56, 3072]}
    Returns: dict of {emotion: [56, 3072]}
    """
    # 28
    n_layers = len(attn_dirs[EMOS6[0]])
    residual_stream = {}
    
    for emo in EMOS6:
        # 初始化56个位置的向量 [56, 3072]
        # Initialize vectors for 56 positions [56, 3072]
        vectors = []
        for layer_idx in range(n_layers):
            # Attention
            vectors.append(attn_dirs[emo][layer_idx])
            # MLP
            vectors.append(mlp_dirs[emo][layer_idx])
        
        # [56, 3072]
        residual_stream[emo] = np.stack(vectors, axis=0)
    
    print(f"\nResidual stream sequence built: 56 positions × 3072 dims")
    return residual_stream

# ============== 相似度计算 / Similarity Computation ==============
def compute_similarity_matrices(residual_stream):
    """
    计算每个情绪的56×56余弦相似度矩阵
    Compute 56×56 cosine similarity matrix for each emotion
    
    返回：dict of {emotion: [56, 56]}
    Returns: dict of {emotion: [56, 56]}
    """
    print("\nComputing cosine similarity matrices...")
    similarity_matrices = {}
    
    for emo in EMOS6:
        # [56, 3072]
        vectors = residual_stream[emo]
        # [56, 56]
        sim_matrix = cosine_similarity(vectors)
        similarity_matrices[emo] = sim_matrix
        
        # 打印统计信息
        # Print statistics
        # 对角线应该全是1，非对角线元素的统计
        # Diagonal should be all 1s, statistics for off-diagonal elements
        mask = ~np.eye(56, dtype=bool)
        off_diag = sim_matrix[mask]
        print(f"  {emo:>10s}: diag={np.diag(sim_matrix).mean():.4f}, "
              f"off-diag mean={off_diag.mean():.4f}, "
              f"off-diag std={off_diag.std():.4f}")
    
    return similarity_matrices

# ============== 可视化 / Visualization ==============
def plot_similarity_heatmap(sim_matrix, emotion, save_path):
    """
    绘制单个情绪的相似度热力图
    Plot similarity heatmap for a single emotion
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 绘制热力图
    # Plot heatmap
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto', 
                   vmin=-1, vmax=1, interpolation='nearest')
    
    # 添加colorbar
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=11)
    
    # 设置标题
    # Set title
    ax.set_title(f'Emotion Direction Similarity: {emotion.upper()}\n'
                 f'Across Residual Stream (56 positions)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 设置坐标轴标签
    # Set axis labels
    ax.set_xlabel('Position in Residual Stream', fontsize=12)
    ax.set_ylabel('Position in Residual Stream', fontsize=12)
    
    # 设置刻度
    # Set ticks
    # 每2个位置（每层）标记一次，显示Attn和MLP
    # Mark every 2 positions (every layer), show Attn and MLP
    # 0, 2, 4, 6, ..., 54
    tick_positions = list(range(0, 56, 2))
    tick_labels = []
    for pos in tick_positions:
        layer = pos // 2
        tick_labels.append(f"L{layer}\nAttn")
    
    # 添加MLP位置
    # Add MLP positions
    # 1, 3, 5, 7, ..., 55
    mlp_positions = list(range(1, 56, 2))
    mlp_labels = []
    for pos in mlp_positions:
        layer = pos // 2
        mlp_labels.append(f"L{layer}\nMLP")
    
    # 合并所有位置和标签
    # Merge all positions and labels
    all_positions = sorted(tick_positions + mlp_positions)
    all_labels = []
    for pos in all_positions:
        layer = pos // 2
        if pos % 2 == 0:
            all_labels.append(f"L{layer}\nAttn")
        else:
            all_labels.append(f"L{layer}\nMLP")
    
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=7, rotation=45)
    ax.set_yticks(all_positions)
    ax.set_yticklabels(all_labels, fontsize=7)
    
    # 添加网格线，每2个位置（每层）一条
    # Add grid lines, one every 2 positions (every layer)
    for i in range(0, 57, 2):
        ax.axhline(i-0.5, color='gray', linewidth=0.3, alpha=0.3)
        ax.axvline(i-0.5, color='gray', linewidth=0.3, alpha=0.3)
    
    # 添加对角线附近的统计信息
    # Add statistics near diagonal
    diag_mean = np.diag(sim_matrix).mean()
    mask = ~np.eye(56, dtype=bool)
    off_diag_mean = sim_matrix[mask].mean()
    off_diag_std = sim_matrix[mask].std()
    
    # 在图上添加文本信息
    # Add text information on plot
    info_text = (f"Diagonal: {diag_mean:.4f}\n"
                 f"Off-diagonal: {off_diag_mean:.4f} ± {off_diag_std:.4f}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")

def plot_local_heatmap(sim_matrix, emotion, layer_start, layer_end, save_path):
    """
    绘制局部层段的热力图
    Plot local layer segment heatmap
    """
    # 计算位置范围
    # Calculate position range
    # Attn位置 / Attn position
    pos_start = layer_start * 2
    # MLP位置 / MLP position
    pos_end = layer_end * 2 + 1
    
    # 提取局部矩阵
    # Extract local matrix
    local_matrix = sim_matrix[pos_start:pos_end+1, pos_start:pos_end+1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    # Plot heatmap
    im = ax.imshow(local_matrix, cmap='RdYlBu_r', aspect='auto', 
                   vmin=-1, vmax=1, interpolation='nearest')
    
    # 添加colorbar
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=11)
    
    # 设置标题
    # Set title
    ax.set_title(f'Emotion Direction Similarity: {emotion.upper()}\n'
                 f'Layers {layer_start}-{layer_end} (Deep Stable Region)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 设置坐标轴标签
    # Set axis labels
    ax.set_xlabel('Position in Residual Stream', fontsize=12)
    ax.set_ylabel('Position in Residual Stream', fontsize=12)
    
    # 设置刻度 - 只显示这个范围内的层
    # Set ticks - only show layers in this range
    local_positions = []
    local_labels = []
    for pos in range(pos_start, pos_end+1):
        layer = pos // 2
        # 相对位置
        # Relative position
        local_positions.append(pos - pos_start)
        if pos % 2 == 0:
            local_labels.append(f"L{layer}\nAttn")
        else:
            local_labels.append(f"L{layer}\nMLP")
    
    ax.set_xticks(range(len(local_positions)))
    ax.set_xticklabels(local_labels, fontsize=8, rotation=45)
    ax.set_yticks(range(len(local_positions)))
    ax.set_yticklabels(local_labels, fontsize=8)
    
    # 添加网格线
    # Add grid lines
    for i in range(0, len(local_positions), 2):
        ax.axhline(i-0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(i-0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    # 添加统计信息
    # Add statistics
    diag_mean = np.diag(local_matrix).mean()
    mask = ~np.eye(len(local_matrix), dtype=bool)
    off_diag_mean = local_matrix[mask].mean()
    off_diag_std = local_matrix[mask].std()
    
    info_text = (f"Layers {layer_start}-{layer_end}\n"
                 f"Diagonal: {diag_mean:.4f}\n"
                 f"Off-diagonal: {off_diag_mean:.4f} ± {off_diag_std:.4f}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")

# ============== 分析函数 / Analysis Functions ==============
def analyze_adjacent_similarity(similarity_matrices):
    """
    分析相邻位置的相似度
    Analyze similarity of adjacent positions
    """
    print("\n" + "=" * 70)
    print("Adjacent Position Similarity Analysis")
    print("=" * 70)
    
    results = {}
    for emo in EMOS6:
        sim_matrix = similarity_matrices[emo]
        
        # 提取对角线上方第1条（相邻位置）
        # Extract first diagonal above main diagonal (adjacent positions)
        # [55]
        adjacent_sim = np.diag(sim_matrix, k=1)
        
        # 区分层内相邻（Attn->MLP）和层间相邻（MLP->Attn）
        # Distinguish within-layer adjacent (Attn->MLP) and between-layer adjacent (MLP->Attn)
        # 偶数位置：Attn->MLP
        # Even positions: Attn->MLP
        within_layer = adjacent_sim[0::2]
        # 奇数位置：MLP->Attn(下一层)
        # Odd positions: MLP->Attn(next layer)
        between_layer = adjacent_sim[1::2]
        
        results[emo] = {
            'within_layer': within_layer,
            'between_layer': between_layer
        }
        
        print(f"\n{emo.upper()}:")
        print(f"  Within layer (Attn→MLP):  mean={within_layer.mean():.4f}, std={within_layer.std():.4f}")
        print(f"  Between layer (MLP→Attn): mean={between_layer.mean():.4f}, std={between_layer.std():.4f}")
    
    return results

def plot_adjacent_similarity_comparison(adjacent_results, save_path):
    """
    绘制相邻位置相似度的对比图
    Plot comparison chart of adjacent position similarity
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, emo in enumerate(EMOS6):
        ax = axes[idx]
        
        within = adjacent_results[emo]['within_layer']
        between = adjacent_results[emo]['between_layer']
        
        # within有28个，between有27个（最后一层MLP后面没有Attn）
        # within has 28, between has 27 (last layer MLP has no Attn after it)
        x_within = np.arange(len(within))
        x_between = np.arange(len(between))
        
        ax.plot(x_within, within, 'o-', label='Within layer (Attn→MLP)', 
                color='steelblue', linewidth=2, markersize=4, alpha=0.7)
        ax.plot(x_between, between, 's-', label='Between layer (MLP→Attn)', 
                color='coral', linewidth=2, markersize=4, alpha=0.7)
        
        ax.set_title(f'{emo.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('Cosine Similarity', fontsize=10)
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
        
        # 添加平均值水平线
        # Add mean value horizontal lines
        ax.axhline(within.mean(), color='steelblue', linestyle='--', 
                   linewidth=1, alpha=0.5)
        ax.axhline(between.mean(), color='coral', linestyle='--', 
                   linewidth=1, alpha=0.5)
    
    plt.suptitle('Adjacent Position Similarity Across Layers', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved adjacent similarity plot: {save_path}")

# ============== 主函数 / Main Function ==============
def main():
    """
    主函数
    Main function
    """
    print("=" * 70)
    print("Emotion Direction Cross-Layer Similarity Analysis")
    print("=" * 70)
    
    # 输出目录
    # Output directory
    out_dir = ensure_dir(PROJECT_ROOT / "outputs" / MODEL_NAME / "06_emotion_circuit_integration" / "emotion_direction_similarity")
    
    # 1. 加载数据
    # 1. Load data
    attn_dirs, mlp_dirs = load_emotion_directions()
    
    # 2. 构建残差流序列
    # 2. Build residual stream sequence
    residual_stream = build_residual_stream_sequence(attn_dirs, mlp_dirs)
    
    # 3. 计算相似度矩阵
    # 3. Compute similarity matrices
    similarity_matrices = compute_similarity_matrices(residual_stream)
    
    # 4. 保存相似度矩阵
    # 4. Save similarity matrices
    print("\nSaving similarity matrices...")
    for emo in EMOS6:
        save_path = out_dir / f"{emo}_similarity_matrix.npy"
        np.save(save_path, similarity_matrices[emo])
        print(f"  Saved: {save_path}")
    
    # 5. 绘制热力图
    # 5. Generate heatmaps
    print("\nGenerating heatmaps...")
    for emo in EMOS6:
        save_path = out_dir / f"{emo}_similarity_heatmap.png"
        plot_similarity_heatmap(similarity_matrices[emo], emo, save_path)
    
    # 5.5. 绘制11-27层局部热力图
    # 5.5. Generate local heatmaps (Layers 11-27)
    print("\nGenerating local heatmaps (Layers 11-27)...")
    for emo in EMOS6:
        save_path = out_dir / f"{emo}_similarity_heatmap_layers11-27.png"
        plot_local_heatmap(similarity_matrices[emo], emo, 11, 27, save_path)
    
    # 6. 分析相邻位置的相似度
    # 6. Analyze adjacent position similarity
    adjacent_results = analyze_adjacent_similarity(similarity_matrices)
    
    # 7. 绘制相邻位置相似度对比图
    # 7. Plot adjacent position similarity comparison
    plot_path = out_dir / "adjacent_similarity_comparison.png"
    plot_adjacent_similarity_comparison(adjacent_results, plot_path)
    
    # 8. 保存统计摘要
    # 8. Save summary statistics
    print("\nSaving summary statistics...")
    summary_path = out_dir / "similarity_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Emotion Direction Similarity Analysis Summary\n")
        f.write("=" * 70 + "\n\n")
        
        for emo in EMOS6:
            sim_matrix = similarity_matrices[emo]
            mask = ~np.eye(56, dtype=bool)
            off_diag = sim_matrix[mask]
            
            f.write(f"{emo.upper()}:\n")
            f.write(f"  Diagonal mean: {np.diag(sim_matrix).mean():.6f}\n")
            f.write(f"  Off-diagonal mean: {off_diag.mean():.6f}\n")
            f.write(f"  Off-diagonal std: {off_diag.std():.6f}\n")
            f.write(f"  Off-diagonal min: {off_diag.min():.6f}\n")
            f.write(f"  Off-diagonal max: {off_diag.max():.6f}\n")
            
            within = adjacent_results[emo]['within_layer']
            between = adjacent_results[emo]['between_layer']
            f.write(f"  Within-layer (Attn→MLP) mean: {within.mean():.6f} ± {within.std():.6f}\n")
            f.write(f"  Between-layer (MLP→Attn) mean: {between.mean():.6f} ± {between.std():.6f}\n")
            f.write("\n")
    
    print(f"  Saved: {summary_path}")
    
    # 总结
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {out_dir}")
    print(f"  - 6 × similarity matrices (.npy)")
    print(f"  - 6 × similarity heatmaps (.png)")
    print(f"  - 6 × local heatmaps for layers 11-27 (.png)")
    print(f"  - 1 × adjacent similarity comparison (.png)")
    print(f"  - 1 × summary statistics (.txt)")

if __name__ == "__main__":
    main()

