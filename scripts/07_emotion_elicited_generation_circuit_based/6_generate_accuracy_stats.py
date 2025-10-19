#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/07_emotion_elicited_generation_circuit_based/6_generate_accuracy_stats.py
"""
全局电路情绪注入准确率统计
Global circuit emotion injection accuracy statistics

功能 Functions:
  1. 读取GPT打标的circuit_labels.jsonl文件
     Read GPT-labeled circuit_labels.jsonl file
  2. 按情绪和极性分类统计匹配率
     Calculate match rates by emotion and valence
  3. 生成详细的准确率报告
     Generate detailed accuracy report

输入输出 Input/Output:
  - 输入 Input:  outputs/{model_name}/07_emotion_elicited_generation_circuit_based/circuit_steered_generation/labeled/circuit_labels.jsonl
  - 输出 Output: outputs/{model_name}/07_emotion_elicited_generation_circuit_based/circuit_steered_generation/labeled/accuracy_stats.json
"""

import json, argparse
from pathlib import Path
from collections import defaultdict

# ============== 路径与基本配置 ==============
# ============== Paths and Basic Configuration ==============
# 项目根目录：自动获取脚本所在位置的上两级目录
# Project root: automatically get the directory two levels up from the script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "llama32_3b"
EMOTIONS = ["anger","sadness","happiness","fear","surprise","disgust"]
VALENCES = ["positive", "neutral", "negative"]

# ============== 统计生成 ==============
# ============== Statistics Generation ==============
def generate_accuracy_stats(labels_file: Path, output_file: Path):
    """
    生成准确率统计
    Generate accuracy statistics
    
    参数 Args:
        labels_file: 打标文件路径 labels file path
        output_file: 输出统计文件路径 output statistics file path
    
    返回 Returns: stats dict
    """
    if not labels_file.exists():
        print(f"[-] 错误 Error: 打标文件不存在 Labels file not found: {labels_file}")
        return None
    
    print(f"[+] 正在读取 Reading {labels_file}...")
    
    # 统计字典
    # Statistics dictionaries
    stats_by_emotion = defaultdict(lambda: {"total": 0, "matched": 0, "unmatched": 0})
    stats_by_valence = defaultdict(lambda: {"total": 0, "matched": 0, "unmatched": 0})
    stats_by_emotion_valence = defaultdict(lambda: defaultdict(lambda: {"total": 0, "matched": 0, "unmatched": 0}))
    
    total_texts = 0
    total_matched = 0
    total_unmatched = 0
    
    # 读取打标数据
    # Read labeled data
    with open(labels_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # 遍历所有极性
                # Iterate through all valences
                for valence in VALENCES:
                    if valence not in record.get("valence_results", {}):
                        continue
                    
                    valence_data = record["valence_results"][valence]
                    
                    # 遍历所有情绪
                    # Iterate through all emotions
                    for emotion in EMOTIONS:
                        if emotion not in valence_data:
                            continue
                        
                        label_result = valence_data[emotion]
                        match = label_result.get("match", 0)
                        
                        # 更新统计
                        # Update statistics
                        total_texts += 1
                        stats_by_emotion[emotion]["total"] += 1
                        stats_by_valence[valence]["total"] += 1
                        stats_by_emotion_valence[emotion][valence]["total"] += 1
                        
                        if match == 1:
                            total_matched += 1
                            stats_by_emotion[emotion]["matched"] += 1
                            stats_by_valence[valence]["matched"] += 1
                            stats_by_emotion_valence[emotion][valence]["matched"] += 1
                        else:
                            total_unmatched += 1
                            stats_by_emotion[emotion]["unmatched"] += 1
                            stats_by_valence[valence]["unmatched"] += 1
                            stats_by_emotion_valence[emotion][valence]["unmatched"] += 1
                            
            except json.JSONDecodeError as e:
                print(f"  [!] 警告 Warning: 第 line {line_num} JSON解析错误 JSON parse error: {e}")
                continue
    
    # 计算准确率
    # Calculate accuracy
    for emotion in stats_by_emotion:
        total = stats_by_emotion[emotion]["total"]
        matched = stats_by_emotion[emotion]["matched"]
        stats_by_emotion[emotion]["accuracy"] = round(matched / total * 100, 2) if total > 0 else 0.0
    
    for valence in stats_by_valence:
        total = stats_by_valence[valence]["total"]
        matched = stats_by_valence[valence]["matched"]
        stats_by_valence[valence]["accuracy"] = round(matched / total * 100, 2) if total > 0 else 0.0
    
    for emotion in stats_by_emotion_valence:
        for valence in stats_by_emotion_valence[emotion]:
            total = stats_by_emotion_valence[emotion][valence]["total"]
            matched = stats_by_emotion_valence[emotion][valence]["matched"]
            stats_by_emotion_valence[emotion][valence]["accuracy"] = round(matched / total * 100, 2) if total > 0 else 0.0
    
    # 总体统计
    # Overall statistics
    overall_accuracy = round(total_matched / total_texts * 100, 2) if total_texts > 0 else 0.0
    
    # 构建统计结果
    # Build statistics result
    stats = {
        "overall": {
            "total_texts": total_texts,
            "matched": total_matched,
            "unmatched": total_unmatched,
            "accuracy": overall_accuracy
        },
        "by_emotion": dict(stats_by_emotion),
        "by_valence": dict(stats_by_valence),
        "by_emotion_valence": {emo: dict(stats_by_emotion_valence[emo]) for emo in stats_by_emotion_valence}
    }
    
    # 保存统计文件
    # Save statistics file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"[+] 统计文件已保存 Statistics file saved: {output_file}")
    
    return stats

# ============== 主函数 ==============
# ============== Main Function ==============
def main():
    """
    主函数
    Main function
    """
    parser = argparse.ArgumentParser(description="全局电路情绪注入准确率统计 Global circuit emotion injection accuracy statistics")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                       help="模型名称 Model name")
    parser.add_argument("--input_filename", type=str, default="circuit_labels.jsonl",
                       help="输入打标文件名 Input labels filename")
    parser.add_argument("--output_filename", type=str, default="accuracy_stats.json",
                       help="输出统计文件名 Output statistics filename")
    args = parser.parse_args()
    
    # 路径配置
    # Path configuration
    base_dir = PROJECT_ROOT / "outputs" / args.model_name / "07_emotion_elicited_generation_circuit_based" / "circuit_steered_generation" / "labeled"
    labels_file = base_dir / args.input_filename
    stats_file = base_dir / args.output_filename
    
    print(f"\n{'='*70}")
    print(f"全局电路情绪注入准确率统计 Global Circuit Emotion Injection Accuracy Statistics")
    print(f"{'='*70}")
    print(f"[配置 CONFIG]")
    print(f"  - 输入文件 Input file: {labels_file}")
    print(f"  - 输出文件 Output file: {stats_file}")
    print(f"  - 情绪类型 Emotion types: {EMOTIONS}")
    print(f"  - 事件极性 Event valences: {VALENCES}")
    print()
    
    # 生成统计
    # Generate statistics
    stats = generate_accuracy_stats(labels_file, stats_file)
    
    if not stats:
        print(f"[-] 统计生成失败 Statistics generation failed")
        return
    
    # 显示统计结果
    # Display statistics results
    print(f"\n{'='*70}")
    print(f"统计结果 Statistics Results")
    print(f"{'='*70}")
    
    print(f"\n📊 总体统计 Overall Statistics:")
    print(f"   总文本数 Total texts: {stats['overall']['total_texts']}")
    print(f"   匹配成功 Matched: {stats['overall']['matched']}")
    print(f"   匹配失败 Unmatched: {stats['overall']['unmatched']}")
    print(f"   总体准确率 Overall accuracy: {stats['overall']['accuracy']}%")
    
    print(f"\n📈 按情绪分类 By Emotion:")
    for emotion in EMOTIONS:
        if emotion in stats['by_emotion']:
            e_stats = stats['by_emotion'][emotion]
            print(f"   {emotion:12s}: {e_stats['matched']:4d}/{e_stats['total']:4d} = {e_stats['accuracy']:6.2f}%")
    
    print(f"\n📊 按极性分类 By Valence:")
    for valence in VALENCES:
        if valence in stats['by_valence']:
            v_stats = stats['by_valence'][valence]
            print(f"   {valence:12s}: {v_stats['matched']:4d}/{v_stats['total']:4d} = {v_stats['accuracy']:6.2f}%")
    
    # 找出表现最好和最差的组合
    # Find best and worst performing combinations
    print(f"\n🏆 情绪×极性组合 Emotion × Valence Combinations:")
    combos = []
    for emotion in stats['by_emotion_valence']:
        for valence in stats['by_emotion_valence'][emotion]:
            combo_stats = stats['by_emotion_valence'][emotion][valence]
            combos.append((emotion, valence, combo_stats['accuracy'], combo_stats['matched'], combo_stats['total']))
    
    # 排序显示前5名和后5名
    # Sort and display top 5 and bottom 5
    combos.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n   表现最好 Top 5:")
    for i, (emo, val, acc, matched, total) in enumerate(combos[:5], 1):
        print(f"   {i}. {emo}-{val}: {matched}/{total} = {acc:.2f}%")
    
    if len(combos) > 5:
        print(f"\n   表现最差 Bottom 5:")
        for i, (emo, val, acc, matched, total) in enumerate(combos[-5:], 1):
            print(f"   {i}. {emo}-{val}: {matched}/{total} = {acc:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"[完成] 统计完成！[Done] Statistics completed!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()


