# scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py
# -*- coding: utf-8 -*-
"""
准确率统计生成脚本
Accuracy Statistics Generation Script

读取脚本2生成的accepted和rejected文件，按情绪和极性分类统计准确率
Reads accepted and rejected files from script 2, calculates accuracy by emotion and valence

- 输入 Input: outputs/{model_name}/01_emotion_elicited_generation_prompt_based/labeled/{dataset_name}/{accepted|rejected}.jsonl
- 输出 Output: outputs/{model_name}/01_emotion_elicited_generation_prompt_based/labeled/{dataset_name}/accuracy_stats.json
"""

import json, argparse
from pathlib import Path
from collections import defaultdict

def generate_accuracy_stats(labeled_dir: Path, dataset_name: str):
    """
    为指定数据集生成准确率统计
    Generate accuracy statistics for specified dataset
    """
    
    dataset_dir = labeled_dir / dataset_name
    accepted_path = dataset_dir / "accepted.jsonl"
    rejected_path = dataset_dir / "rejected.jsonl"
    stats_path = dataset_dir / "accuracy_stats.json"
    
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return None
    
    # 统计字典
    # Statistics dictionaries
    stats_by_emotion = defaultdict(lambda: {"total": 0, "accepted": 0, "rejected": 0})
    stats_by_valence = defaultdict(lambda: {"total": 0, "accepted": 0, "rejected": 0})
    
    total_accepted = 0
    total_rejected = 0
    
    # 读取accepted
    # Read accepted
    if accepted_path.exists():
        print(f"Reading {accepted_path}...")
        with open(accepted_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    emotion = item.get("emotion", "unknown")
                    valence = item.get("valence", "unknown")
                    
                    stats_by_emotion[emotion]["total"] += 1
                    stats_by_emotion[emotion]["accepted"] += 1
                    
                    stats_by_valence[valence]["total"] += 1
                    stats_by_valence[valence]["accepted"] += 1
                    
                    total_accepted += 1
                except json.JSONDecodeError:
                    continue
    else:
        print(f"[WARNING] Accepted file not found: {accepted_path}")
    
    # 读取rejected
    # Read rejected
    if rejected_path.exists():
        print(f"Reading {rejected_path}...")
        with open(rejected_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    emotion = item.get("emotion", "unknown")
                    valence = item.get("valence", "unknown")
                    
                    stats_by_emotion[emotion]["total"] += 1
                    stats_by_emotion[emotion]["rejected"] += 1
                    
                    stats_by_valence[valence]["total"] += 1
                    stats_by_valence[valence]["rejected"] += 1
                    
                    total_rejected += 1
                except json.JSONDecodeError:
                    continue
    else:
        print(f"[WARNING] Rejected file not found: {rejected_path}")
    
    # 计算准确率
    # Calculate accuracy
    for emotion in stats_by_emotion:
        total = stats_by_emotion[emotion]["total"]
        accepted = stats_by_emotion[emotion]["accepted"]
        stats_by_emotion[emotion]["accuracy"] = round(accepted / total * 100, 2) if total > 0 else 0.0
    
    for valence in stats_by_valence:
        total = stats_by_valence[valence]["total"]
        accepted = stats_by_valence[valence]["accepted"]
        stats_by_valence[valence]["accuracy"] = round(accepted / total * 100, 2) if total > 0 else 0.0
    
    # 总体统计
    # Overall statistics
    total_samples = total_accepted + total_rejected
    overall_accuracy = round(total_accepted / total_samples * 100, 2) if total_samples > 0 else 0.0
    
    # 构建统计结果
    # Build statistics result
    stats = {
        "dataset": dataset_name,
        "overall": {
            "total_samples": total_samples,
            "accepted": total_accepted,
            "rejected": total_rejected,
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
    parser.add_argument("--input_dir", type=str, default=None,
                       help="labeled目录路径 Labeled directory path，如 outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled")
    parser.add_argument("--both", action="store_true",
                       help="处理sev和test_set两个数据集 Process both sev and test_set datasets")
    parser.add_argument("--model_name", type=str, default="llama32_3b",
                       help="模型文件夹名 Model folder name")
    parser.add_argument("--dataset", type=str, default=None,
                       help="数据集名称 Dataset name (如 sev, test_set)，不指定则处理所有数据集")
    args = parser.parse_args()
    
    # 确定labeled目录路径
    # Determine labeled directory path
    if args.both or (not args.input_dir and not args.dataset):
        labeled_dir = Path("outputs") / args.model_name / "01_emotion_elicited_generation_prompt_based" / "labeled"
    elif args.input_dir:
        labeled_dir = Path(args.input_dir)
    else:
        parser.error("必须指定 --input_dir 或 --both | Must specify --input_dir or --both")
        return
    
    if not labeled_dir.exists():
        print(f"[ERROR] Labeled directory not found: {labeled_dir}")
        return
    
    # 确定要处理的数据集
    # Determine datasets to process
    if args.both:
        datasets = ["sev", "test_set"]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        # 自动发现所有数据集文件夹
        # Auto-discover all dataset folders
        datasets = [d.name for d in labeled_dir.iterdir() if d.is_dir()]
    
    if not datasets:
        print(f"[ERROR] No datasets found in {labeled_dir}")
        return
    
    print("=" * 70)
    print("生成准确率统计文件 | Generating Accuracy Statistics")
    print("=" * 70)
    
    for dataset_name in datasets:
        result = generate_accuracy_stats(labeled_dir, dataset_name)
        
        if result:
            stats, stats_path = result
            
            print(f"\n📊 {dataset_name.upper()} 数据集统计:")
            print(f"   文件 File: {stats_path}")
            print(f"   总样本 Total: {stats['overall']['total_samples']}")
            print(f"   Accepted: {stats['overall']['accepted']}")
            print(f"   Rejected: {stats['overall']['rejected']}")
            print(f"   总体准确率 Overall Accuracy: {stats['overall']['accuracy']}%")
            
            print(f"\n   按情绪分类 By Emotion:")
            for emotion in sorted(stats['by_emotion'].keys()):
                e_stats = stats['by_emotion'][emotion]
                print(f"      {emotion:12s}: {e_stats['accepted']:4d}/{e_stats['total']:4d} = {e_stats['accuracy']:6.2f}%")
            
            if stats['by_valence']:
                print(f"\n   按极性分类 By Valence:")
                for valence in sorted(stats['by_valence'].keys()):
                    v_stats = stats['by_valence'][valence]
                    print(f"      {valence:12s}: {v_stats['accepted']:4d}/{v_stats['total']:4d} = {v_stats['accuracy']:6.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ 统计文件生成完成! | Statistics generation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

