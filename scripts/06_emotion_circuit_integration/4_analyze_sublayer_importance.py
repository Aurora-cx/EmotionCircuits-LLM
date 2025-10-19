# -*- coding: utf-8 -*-
# scripts/06_emotion_circuit_integration/4_analyze_sublayer_importance.py
"""
分析子层重要性：对多个alpha值取平均并归一化为概率分布
Analyze sublayer importance: average across alpha values and normalize to probability distribution

处理流程 Processing flow:
  1. 读取所有alpha值的重要性数据 / Read importance data for all alpha values
  2. 按 (emotion, kind, layer) 分组求平均 / Group by (emotion, kind, layer) and average
  3. 按情绪归一化为概率分布 / Normalize to probability distribution per emotion

输入输出 Input/Output:
  - 输入 Input:  outputs/{model_name}/06_emotion_circuit_integration/sublayer_importance/importance_all_alphas.csv
  - 输出 Output: outputs/{model_name}/06_emotion_circuit_integration/importance_all_alpha_mean_emotion_all_probability.csv
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ============== 路径配置 ==============
# ============== Path Configuration ==============
# 项目根目录：自动获取脚本所在位置的上两级目录
# Project root: automatically get the directory two levels up from the script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL_NAME = "llama32_3b"

# ============== 工具函数 ==============
# ============== Utility Functions ==============
def ensure_dir(p):
    """
    确保目录存在
    Ensure directory exists
    """
    os.makedirs(p, exist_ok=True)
    return p

# ============== 核心处理 ==============
# ============== Core Processing ==============
def compute_alpha_mean_and_normalize(input_csv, output_csv):
    """
    计算多个alpha的平均值并归一化为概率分布
    Compute mean across alpha values and normalize to probability distribution
    
    参数 Args:
        input_csv: 输入CSV文件路径（包含所有alpha值的数据）
                   Input CSV file path (data for all alpha values)
        output_csv: 输出CSV文件路径
                    Output CSV file path
    """
    print(f"[+] 读取数据 Reading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 显示基本信息
    # Display basic info
    print(f"    - 总行数 Total rows: {len(df)}")
    print(f"    - Alpha 值 Alpha values: {sorted(df['alpha'].unique())}")
    print(f"    - 情绪 Emotions: {sorted(df['emotion'].unique())}")
    
    # 步骤1: 按 (emotion, kind, layer) 分组，对 I_final 取平均
    # Step 1: Group by (emotion, kind, layer) and average I_final
    print(f"\n[+] 计算 alpha 平均值 Computing alpha mean...")
    df_mean = df.groupby(['emotion', 'kind', 'layer']).agg({
        'I_final': 'mean'
    }).reset_index()
    
    print(f"    - 平均后行数 Rows after averaging: {len(df_mean)}")
    
    # 步骤2: 按情绪归一化为概率分布
    # Step 2: Normalize to probability distribution per emotion
    print(f"\n[+] 归一化为概率分布 Normalizing to probability distribution...")
    
    df_mean['I_final_normalized'] = 0.0
    
    for emotion in df_mean['emotion'].unique():
        # 获取该情绪的所有行
        # Get all rows for this emotion
        mask = df_mean['emotion'] == emotion
        
        # 计算总和（用于归一化）
        # Compute sum (for normalization)
        total = df_mean.loc[mask, 'I_final'].sum()
        
        # 归一化
        # Normalize
        if total != 0:
            df_mean.loc[mask, 'I_final_normalized'] = df_mean.loc[mask, 'I_final'] / total
        else:
            print(f"    [!] 警告 Warning: {emotion} 的总和为0，跳过归一化 sum is 0, skipping normalization")
        
        # 显示统计信息
        # Display statistics
        normalized_sum = df_mean.loc[mask, 'I_final_normalized'].sum()
        print(f"    - {emotion}: I_final 总和 sum={total:.6f}, 归一化后总和 normalized sum={normalized_sum:.6f}")
    
    # 保存结果
    # Save results
    print(f"\n[+] 保存结果 Saving results to: {output_csv}")
    df_mean.to_csv(output_csv, index=False)
    print(f"[✓] 完成 Done! 生成 {len(df_mean)} 行数据 Generated {len(df_mean)} rows")
    
    # 显示样例数据
    # Display sample data
    print(f"\n[+] 样例数据 Sample data (前10行 first 10 rows):")
    print(df_mean.head(10).to_string(index=False))
    
    return df_mean

# ============== 主函数 ==============
# ============== Main Function ==============
def main():
    """
    主函数
    Main function
    """
    ap = argparse.ArgumentParser(description="分析子层重要性：alpha平均与概率归一化 Analyze sublayer importance: alpha averaging & probability normalization")
    ap.add_argument("--model_name", default=MODEL_NAME, 
                    help="模型名称 Model name")
    args = ap.parse_args()
    
    # 路径配置
    # Path configuration
    base_dir = PROJECT_ROOT / "outputs" / args.model_name
    dir_06 = base_dir / "06_emotion_circuit_integration"
    dir_importance = dir_06 / "sublayer_importance"
    
    # 输入输出文件
    # Input/Output files
    input_csv = dir_importance / "importance_all_alphas.csv"
    output_csv = dir_06 / "importance_all_alpha_mean_emotion_all_probability.csv"
    
    # 检查输入文件
    # Check input file
    if not input_csv.exists():
        print(f"[-] 错误 Error: 输入文件不存在 Input file does not exist: {input_csv}")
        print(f"    请先运行 Please run first: 3_compute_sublayer_importance_multi_alpha.py")
        return
    
    print(f"\n{'='*70}")
    print(f"分析子层重要性 Analyzing Sublayer Importance")
    print(f"{'='*70}\n")
    
    # 执行处理
    # Execute processing
    df_result = compute_alpha_mean_and_normalize(str(input_csv), str(output_csv))
    
    # 统计摘要
    # Statistical summary
    print(f"\n{'='*70}")
    print(f"统计摘要 Statistical Summary")
    print(f"{'='*70}")
    
    for emotion in sorted(df_result['emotion'].unique()):
        mask = df_result['emotion'] == emotion
        emotion_data = df_result.loc[mask]
        
        print(f"\n{emotion.upper()}:")
        print(f"  - I_final 平均值 mean: {emotion_data['I_final'].mean():.6f}")
        print(f"  - I_final 标准差 std: {emotion_data['I_final'].std():.6f}")
        print(f"  - I_final 最大值 max: {emotion_data['I_final'].max():.6f} @ {emotion_data.loc[emotion_data['I_final'].idxmax(), 'kind']} L{emotion_data.loc[emotion_data['I_final'].idxmax(), 'layer']}")
        print(f"  - I_final 最小值 min: {emotion_data['I_final'].min():.6f} @ {emotion_data.loc[emotion_data['I_final'].idxmin(), 'kind']} L{emotion_data.loc[emotion_data['I_final'].idxmin(), 'layer']}")
    
    print(f"\n{'='*70}")
    print(f"[完成] 分析完成 [Done] Analysis completed!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

