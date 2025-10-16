# scripts/03_emotion_elicited_generation_steer_based/3_generate_accuracy_stats.py
# -*- coding: utf-8 -*-
"""
情绪引导生成准确率统计脚本
Emotion-Steered Generation Accuracy Statistics Script

读取GPT打标结果，按情绪和极性分类统计准确率
Reads GPT labeling results, calculates accuracy by emotion and valence

输入 Input: outputs/{model_name}/03_emotion_steered_generation/{dataset_name}/labeled_results.jsonl
输出 Output: outputs/{model_name}/03_emotion_steered_generation/{dataset_name}/accuracy_stats.json
"""

import json, argparse
from pathlib import Path
from collections import defaultdict

def generate_accuracy_stats(output_dir: Path, dataset_name: str):
    """
    为指定数据集生成准确率统计
    Generate accuracy statistics for specified dataset
    """
    
    dataset_dir = output_dir / dataset_name
    labeled_path = dataset_dir / "labeled_results.jsonl"
    stats_path = dataset_dir / "accuracy_stats.json"
    
    if not labeled_path.exists():
        print(f"[ERROR] Labeled results file not found: {labeled_path}")
        return None
    
    # 统计字典
    # Statistics dictionaries
    stats_by_emotion = defaultdict(lambda: {"total": 0, "matches": 0})
    stats_by_valence = defaultdict(lambda: {"total": 0, "matches": 0})
    
    total_pairs = 0
    total_matches = 0
    
    # 读取labeled_results
    # Read labeled_results
    print(f"Reading {labeled_path}...")
    with open(labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                
                # valence_labels 包含 positive, neutral, negative 三个键
                # valence_labels contains positive, neutral, negative keys
                valence_labels = item.get("valence_labels", {})
                
                # 遍历每个极性
                # Iterate through each valence
                for event_valence, emotion_labels in valence_labels.items():
                    # 统计每个情绪的匹配结果
                    # Count matching results for each emotion
                    for emotion, label in emotion_labels.items():
                        match = label.get("match", 0)
                        
                        stats_by_emotion[emotion]["total"] += 1
                        stats_by_valence[event_valence]["total"] += 1
                        total_pairs += 1
                        
                        if match == 1:
                            stats_by_emotion[emotion]["matches"] += 1
                            stats_by_valence[event_valence]["matches"] += 1
                            total_matches += 1
                        
            except json.JSONDecodeError:
                continue
    
    # 计算准确率
    # Calculate accuracy
    for emotion in stats_by_emotion:
        total = stats_by_emotion[emotion]["total"]
        matches = stats_by_emotion[emotion]["matches"]
        stats_by_emotion[emotion]["accuracy"] = round(matches / total * 100, 2) if total > 0 else 0.0
    
    for valence in stats_by_valence:
        total = stats_by_valence[valence]["total"]
        matches = stats_by_valence[valence]["matches"]
        stats_by_valence[valence]["accuracy"] = round(matches / total * 100, 2) if total > 0 else 0.0
    
    # 总体统计
    # Overall statistics
    overall_accuracy = round(total_matches / total_pairs * 100, 2) if total_pairs > 0 else 0.0
    
    # 构建统计结果
    # Build statistics result
    stats = {
        "dataset": dataset_name,
        "overall": {
            "total_pairs": total_pairs,
            "matches": total_matches,
            "accuracy": overall_accuracy
        },
        "by_emotion": dict(stats_by_emotion),
        "by_valence": dict(stats_by_valence)
    }
    
    # 保存统计文件
    # Save statistics file
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats, stats_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama32_3b",
                       help="模型文件夹名称 Model folder name")
    parser.add_argument("--dataset", type=str, default="test_set",
                       help="数据集名称 Dataset name (如 test_set, user_test)，不指定则处理所有数据集")
    parser.add_argument("--all", action="store_true",
                       help="处理所有数据集 Process all datasets")
    args = parser.parse_args()
    
    # 确定输出目录路径
    # Determine output directory path
    output_dir = Path("outputs") / args.model_name / "03_emotion_steered_generation"
    
    if not output_dir.exists():
        print(f"[ERROR] Output directory not found: {output_dir}")
        return
    
    # 确定要处理的数据集
    # Determine datasets to process
    if args.all:
        # 自动发现所有数据集文件夹
        # Auto-discover all dataset folders
        datasets = [d.name for d in output_dir.iterdir() if d.is_dir() and (d / "labeled_results.jsonl").exists()]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        # 默认处理所有包含labeled_results.jsonl的数据集
        # Default: process all datasets with labeled_results.jsonl
        datasets = [d.name for d in output_dir.iterdir() if d.is_dir() and (d / "labeled_results.jsonl").exists()]
    
    if not datasets:
        print(f"[ERROR] No datasets with labeled_results.jsonl found in {output_dir}")
        return
    
    print("=" * 70)
    print("生成情绪引导生成准确率统计 | Generating Emotion Steering Accuracy Stats")
    print("=" * 70)
    
    for dataset_name in datasets:
        result = generate_accuracy_stats(output_dir, dataset_name)
        
        if result:
            stats, stats_path = result
            
            print(f"\n📊 {dataset_name.upper()} 数据集统计 Dataset Statistics:")
            print(f"   文件 File: {stats_path}")
            print(f"   总配对数 Total Pairs: {stats['overall']['total_pairs']}")
            print(f"   匹配数 Matches: {stats['overall']['matches']}")
            print(f"   总体准确率 Overall Accuracy: {stats['overall']['accuracy']}%")
            
            print(f"\n   按情绪分类 By Emotion:")
            for emotion in sorted(stats['by_emotion'].keys()):
                e_stats = stats['by_emotion'][emotion]
                print(f"      {emotion:12s}: {e_stats['matches']:4d}/{e_stats['total']:4d} = {e_stats['accuracy']:6.2f}%")
            
            if stats['by_valence']:
                print(f"\n   按极性分类 By Valence:")
                for valence in sorted(stats['by_valence'].keys()):
                    v_stats = stats['by_valence'][valence]
                    print(f"      {valence:12s}: {v_stats['matches']:4d}/{v_stats['total']:4d} = {v_stats['accuracy']:6.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ 统计文件生成完成! | Statistics generation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

