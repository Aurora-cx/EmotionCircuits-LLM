#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/07_emotion_elicited_generation_circuit_based/2_visualize_global_circuit.py
"""
全局电路情绪注入残差流可视化脚本
Global circuit emotion injection residual stream visualization script

功能 Functions:
  1. 加载全局电路的情绪注入残差流数据
     Load global circuit emotion injection residual stream data
  2. 对每层的MLP/Attention hidden states进行降维可视化
     Dimensionality reduction and visualization of MLP/Attention hidden states per layer
  3. 展示情绪在全局电路调控下的聚簇分离情况
     Display emotion cluster separation under global circuit regulation

输入输出 Input/Output:
  - 输入 Input:
    - outputs/{model_name}/07_emotion_elicited_generation_circuit_based/global_circuit_residuals/circuit_residual_streams_outputs.jsonl - 元数据 / Metadata
    - outputs/{model_name}/07_emotion_elicited_generation_circuit_based/global_circuit_residuals/*_residuals.npz - 残差流数据文件 / Residual stream data files
  - 输出 Output:
    - outputs/{model_name}/07_emotion_elicited_generation_circuit_based/global_circuit_residuals/visualizations/ - 每层的聚簇可视化图片 / Cluster visualization images per layer
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd

# ============== 路径与基本配置 ==============
# ============== Paths and Basic Configuration ==============
# 项目根目录：自动获取脚本所在位置的上两级目录
# Project root: automatically get the directory two levels up from the script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL_NAME = "llama32_3b"
EMOS6 = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]
VALENCES = ["positive", "neutral", "negative"]

# ============== 工具函数 ==============
# ============== Utility Functions ==============
def ensure_dir(path):
    """
    确保目录存在
    Ensure directory exists
    """
    os.makedirs(path, exist_ok=True)
    return path

# ============== 数据加载 ==============
# ============== Data Loading ==============
def load_global_circuit_data(model_name, max_samples=None):
    """
    加载全局电路情绪注入残差流数据
    Load global circuit emotion injection residual stream data
    
    参数 Args:
        model_name: 模型名称 model name
        max_samples: 最大样本数（可选）max samples (optional)
    
    返回 Returns: 包含残差流数据的列表 list containing residual stream data
    """
    residual_dir = PROJECT_ROOT / "outputs" / model_name / "07_emotion_elicited_generation_circuit_based" / "global_circuit_residuals"
    json_path = residual_dir / "circuit_residual_streams_outputs.jsonl"
    
    if not json_path.exists():
        print(f"[-] 错误 Error: Global circuit data not found at {json_path}")
        return []
    
    # 读取元数据
    # Read metadata
    metadata = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            metadata.append(data)
    
    print(f"[+] 已加载 Loaded {len(metadata)} global circuit metadata entries")
    
    # 加载残差流数据
    # Load residual stream data
    all_data = []
    
    for meta in tqdm(metadata[:max_samples] if max_samples else metadata, desc="加载残差流数据 Loading residual data"):
        skeleton_id = meta['skeleton_id']
        
        for valence in VALENCES:
            if valence not in meta['valence_results']:
                continue
                
            for emotion in EMOS6:
                if emotion not in meta['valence_results'][valence]:
                    continue
                    
                result = meta['valence_results'][valence][emotion]
                if 'residual_file' in result:
                    residual_file = residual_dir / result['residual_file']
                    if residual_file.exists():
                        try:
                            data = np.load(residual_file)
                            all_data.append({
                                'skeleton_id': skeleton_id,
                                'valence': valence,
                                'emotion': emotion,
                                'residual_data': data
                            })
                        except Exception as e:
                            print(f"  [!] 加载错误 Error loading {residual_file}: {e}")
    
    print(f"[+] 已加载 Loaded {len(all_data)} global circuit residual data entries")
    return all_data

def extract_layer_data(all_data, layer_idx, component='mlp'):
    """
    提取指定层的所有样本数据
    Extract data for all samples at specified layer
    
    参数 Args:
        all_data: 所有数据列表 all data list
        layer_idx: 层索引 layer index
        component: 组件类型（'mlp' 或 'attention'）component type ('mlp' or 'attention')
    
    返回 Returns: layer_data (numpy array), labels (list)
    """
    layer_data = []
    labels = []
    
    for item in all_data:
        residual_data = item['residual_data']
        
        # 构建键名
        # Build key name
        if component == 'attention':
            key = f"attention_{layer_idx}"
        else:  # mlp
            key = f"mlp_{layer_idx}"
            
        if key in residual_data:
            hidden_state = residual_data[key]  # (1, 3072)
            layer_data.append(hidden_state.squeeze())  # 转换为 (3072,) / convert to (3072,)
            labels.append({
                'emotion': item['emotion'],
                'valence': item['valence'],
                'skeleton_id': item['skeleton_id']
            })
    
    if len(layer_data) == 0:
        return None, None
        
    return np.array(layer_data), labels

# ============== 降维处理 ==============
# ============== Dimensionality Reduction ==============
def reduce_dimensions(data, method='pca', n_components=2):
    """
    降维处理
    Dimensionality reduction
    
    参数 Args:
        data: 输入数据 input data
        method: 降维方法（'pca' 或 'tsne'）reduction method ('pca' or 'tsne')
        n_components: 目标维度 target dimensions
    
    返回 Returns: 降维后的数据 reduced data, 降维器 reducer
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'tsne':
        # 先用PCA降维到50维，再用t-SNE
        # First reduce to 50 dimensions with PCA, then apply t-SNE
        pca = PCA(n_components=min(50, data.shape[1]), random_state=42)
        data_pca = pca.fit_transform(data)
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        return reducer.fit_transform(data_pca), reducer
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(data), reducer

# ============== 可视化 ==============
# ============== Visualization ==============
def plot_emotion_clusters(data_2d, labels, title, save_path, method='pca'):
    """
    绘制情绪聚簇图
    Plot emotion cluster visualization
    
    参数 Args:
        data_2d: 2维数据 2D data
        labels: 标签列表 label list
        title: 图标题 plot title
        save_path: 保存路径 save path
        method: 降维方法 reduction method
    """
    # 创建DataFrame
    # Create DataFrame
    df = pd.DataFrame({
        'x': data_2d[:, 0],
        'y': data_2d[:, 1],
        'emotion': [label['emotion'] for label in labels],
        'valence': [label['valence'] for label in labels]
    })
    
    # 设置图形
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # 情绪颜色映射
    # Emotion color mapping
    emotion_colors = {
        'anger': '#FF6B6B',
        'sadness': '#4ECDC4', 
        'happiness': '#45B7D1',
        'fear': '#96CEB4',
        'disgust': '#FFEAA7',
        'surprise': '#DDA0DD'
    }
    
    # 绘制散点图
    # Plot scatter points
    for emotion in EMOS6:
        emotion_data = df[df['emotion'] == emotion]
        if len(emotion_data) > 0:
            plt.scatter(emotion_data['x'], emotion_data['y'], 
                       c=emotion_colors[emotion], 
                       label=emotion, 
                       alpha=0.7, 
                       s=50,
                       edgecolors='black',
                       linewidth=0.1)
    
    # 设置样式
    # Set style
    plt.xlabel('')
    plt.ylabel('')
    plt.tick_params(axis='both', which='major', labelsize=25)  # 设置刻度字体大小 / set tick font size
    
    # 设置整个图框的边框粗细
    # Set border thickness
    for spine in plt.gca().spines.values():
        spine.set_linewidth(3.0)
    
    plt.grid(True, alpha=0.3)
    
    # 移除标题
    # Remove title
    plt.title('')
    
    # 添加水平legend（仅对第0层）
    # Add horizontal legend (only for layer 0)
    if 'L0' in str(save_path):
        plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============== 主函数 ==============
# ============== Main Function ==============
def main():
    """
    主函数
    Main function
    """
    parser = argparse.ArgumentParser(description="全局电路情绪注入残差流可视化 Visualize global circuit emotion injection residual streams")
    parser.add_argument("--model_name", default=MODEL_NAME,
                        help="模型名称 Model name")
    parser.add_argument("--method", default="pca", choices=["pca", "tsne"],
                        help="降维方法 Dimensionality reduction method")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数 Maximum number of samples to process")
    parser.add_argument("--layers", default="1,9,12,27",
                        help="要处理的层（例如 Layers to process (e.g., '1,9,12,27' or 'all')")
    parser.add_argument("--component", default="mlp", choices=["mlp", "attention", "both"],
                        help="要可视化的组件 Component to visualize")
    parser.add_argument("--format", default="pdf", choices=["png", "pdf"],
                        help="输出图片格式 Output image format")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"全局电路情绪注入可视化 Global Circuit Emotion Injection Visualization")
    print(f"{'='*70}")
    print(f"[配置 CONFIG]")
    print(f"  - 模型 Model: {args.model_name}")
    print(f"  - 降维方法 Method: {args.method.upper()}")
    print(f"  - 最大样本数 Max samples: {args.max_samples or 'All'}")
    print(f"  - 组件 Component: {args.component}")
    print(f"  - 输出格式 Format: {args.format}")
    print()
    
    # 加载数据
    # Load data
    print(f"[+] 正在加载全局电路残差流数据 Loading global circuit residual data...")
    all_data = load_global_circuit_data(args.model_name, max_samples=args.max_samples)
    
    if len(all_data) == 0:
        print(f"[-] 未加载到数据，退出 No data loaded. Exiting.")
        return
    
    # 创建输出目录
    # Create output directory
    output_dir = PROJECT_ROOT / "outputs" / args.model_name / "07_emotion_elicited_generation_circuit_based" / "global_circuit_residuals" / "visualizations"
    ensure_dir(output_dir)
    print(f"[+] 输出目录 Output directory: {output_dir}")
    
    # 确定要处理的层
    # Determine layers to process
    if args.layers == "all":
        layers_to_process = list(range(28))
    else:
        layers_to_process = [int(x.strip()) for x in args.layers.split(",")]
    
    print(f"[+] 处理层 Processing layers: {layers_to_process}")
    
    # 确定要处理的组件
    # Determine components to process
    if args.component == "both":
        components = ["mlp", "attention"]
    else:
        components = [args.component]
    
    print(f"[+] 处理组件 Processing components: {components}")
    print()
    
    # 处理每层每个组件
    # Process each layer and component
    for component in components:
        print(f"\n{'='*70}")
        print(f"处理组件 Processing component: {component.upper()}")
        print(f"{'='*70}\n")
        
        for layer_idx in tqdm(layers_to_process, desc=f"处理层 Processing {component} layers"):
            print(f"\n[+] 处理 Processing Layer {layer_idx} {component.upper()}...")
            
            # 提取该层数据
            # Extract layer data
            layer_data, layer_labels = extract_layer_data(all_data, layer_idx, component)
            
            if layer_data is None or len(layer_data) < 10:
                n_samples = len(layer_data) if layer_data is not None else 0
                print(f"  [-] 跳过 Skipping Layer {layer_idx} {component} - 数据不足 insufficient data ({n_samples} samples)")
                continue
            
            print(f"  [+] 已提取 Extracted {len(layer_data)} samples")
            
            # 标准化数据
            # Standardize data
            scaler = StandardScaler()
            layer_data_scaled = scaler.fit_transform(layer_data)
            
            # 降维
            # Reduce dimensions
            print(f"  [+] 应用 Applying {args.method.upper()}...")
            data_2d, reducer = reduce_dimensions(layer_data_scaled, method=args.method)
            
            # 绘制情绪聚簇图
            # Plot emotion clusters
            emotion_title = f"Layer {layer_idx} {component.upper()} - Emotion Clusters"
            emotion_save_path = output_dir / f"L{layer_idx}_{component}_emotion_{args.method}.{args.format}"
            plot_emotion_clusters(data_2d, layer_labels, emotion_title, str(emotion_save_path), args.method)
            print(f"  [✓] 已保存情绪聚簇图 Saved emotion clusters: {emotion_save_path}")
    
    print(f"\n{'='*70}")
    print(f"[完成] 全部可视化已保存到 [Done] All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

