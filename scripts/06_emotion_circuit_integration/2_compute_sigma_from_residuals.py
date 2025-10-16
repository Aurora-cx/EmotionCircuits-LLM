#!/usr/bin/env python
# -*- coding: utf-8 -*-
# scripts/06_emotion_circuit_integration/2_compute_sigma_from_residuals.py
"""
批量计算各层残差流的标准差（RMS）
Batch compute RMS (standard deviation) of residual stream for each layer

从 outputs/{model_name}/02_emotion_directions/residual_dump/{attention,mlp}/ 下的 .pt 文件计算sigma
Compute sigma from .pt files under outputs/{model_name}/02_emotion_directions/residual_dump/{attention,mlp}/

- 输入 Input: outputs/{model_name}/02_emotion_directions/residual_dump/{attention,mlp}/*.pt
- 输出 Output: outputs/{model_name}/06_emotion_circuit_integration/sigma_summary.json
"""

import torch
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ============== 路径与基本配置 / Paths and Basic Configuration ==============
# 工作目录：项目根目录
# Working directory: project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL_NAME = 'llama32_3b'
SUBDIRS = ['attention', 'mlp']
EMOTIONS = ['anger', 'sadness', 'happiness', 'fear', 'surprise', 'disgust']

# ============== 主函数 / Main Function ==============
def main():
    """
    主函数
    Main function
    """
    parser = argparse.ArgumentParser(
        description="计算残差流标准差 / Compute residual stream standard deviation"
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="模型名称 / Model name"
    )
    args = parser.parse_args()
    
    # 路径设置
    # Path setup
    # 输入路径：从02_emotion_directions读取残差数据
    # Input path: read residual data from 02_emotion_directions
    input_dir = PROJECT_ROOT / 'outputs' / args.model_name / '02_emotion_directions'
    residual_dir = input_dir / 'residual_dump'
    
    # 输出路径：保存到06_emotion_circuit_integration
    # Output path: save to 06_emotion_circuit_integration
    output_dir = PROJECT_ROOT / 'outputs' / args.model_name / '06_emotion_circuit_integration'
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入目录
    # Check input directory
    if not residual_dir.exists():
        print(f"[-] Error: Directory {residual_dir} does not exist!")
        return
    
    # 存储结果：{('attention', layer_id): [sigma_values], ('mlp', layer_id): [sigma_values]}
    # Store results: {('attention', layer_id): [sigma_values], ('mlp', layer_id): [sigma_values]}
    sigma_dict = defaultdict(list)
    
    for subdir in SUBDIRS:
        data_dir = residual_dir / subdir
        
        # 检查子目录
        # Check subdirectory
        if not data_dir.exists():
            print(f"[-] Warning: Directory {data_dir} does not exist, skipping...")
            continue
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        print(f'\n>>> Processing {subdir}: found {len(files)} files')
        
        for fname in tqdm(files, desc=f"Computing sigma for {subdir}"):
            fpath = data_dir / fname
            
            try:
                data = torch.load(fpath, map_location='cpu', weights_only=False)
                
                # hidden_last_all_layers: {emotion: tensor [28, 3072]}
                hidden_data = data.get('hidden_last_all_layers', {})
                
                if not hidden_data:
                    print(f"  [Warning] No hidden_last_all_layers in {fname}, skipping...")
                    continue
                
                for emotion in EMOTIONS:
                    if emotion not in hidden_data:
                        continue
                    
                    # [28, 3072]
                    tensor = hidden_data[emotion]
                    
                    # 对每一层计算RMS (Root Mean Square)
                    # Compute RMS for each layer
                    # [28]
                    rms = torch.sqrt((tensor ** 2).mean(dim=1))
                    
                    for layer_id, val in enumerate(rms.tolist()):
                        sigma_dict[(subdir, layer_id)].append(val)
                        
            except Exception as e:
                print(f"  [Error] Failed to process {fname}: {e}")
                continue
    
    # 汇总：对每层取所有样本的均值
    # Summary: take mean of all samples for each layer
    sigma_summary = {}
    for key, values in sigma_dict.items():
        subdir, layer_id = key
        if len(values) > 0:
            sigma_mean = sum(values) / len(values)
            sigma_summary[f'{subdir}_{layer_id}'] = sigma_mean
    
    # 打印统计信息
    # Print statistics
    print(f"\n{'='*60}")
    print("Sigma Summary Statistics")
    print(f"{'='*60}")
    
    for subdir in SUBDIRS:
        print(f"\n{subdir.upper()}:")
        for layer_id in range(28):
            key = f'{subdir}_{layer_id}'
            if key in sigma_summary:
                sigma = sigma_summary[key]
                n_samples = len(sigma_dict[(subdir, layer_id)])
                print(f"  Layer {layer_id:2d}: σ={sigma:.6f} (n={n_samples})")
    
    # 保存结果
    # Save results
    output_path = output_dir / 'sigma_summary.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sigma_summary, f, indent=2, ensure_ascii=False)
    
    print(f'\n✅ Done! Sigma summary saved to: {output_path}')
    print(f'   Total entries: {len(sigma_summary)}')

if __name__ == "__main__":
    main()