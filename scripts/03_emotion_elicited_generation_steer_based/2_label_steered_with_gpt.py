# scripts/03_emotion_elicited_generation_steer_based/2_label_steered_with_gpt.py
# -*- coding: utf-8 -*-
"""
情绪引导生成文本的GPT打标脚本
GPT Labeling Script for Emotion-Steered Generated Text

使用GPT-4o-mini判断生成的文本是否符合目标情绪
Use GPT-4o-mini to judge if generated text matches target emotion

输入 Input: outputs/{model_name}/03_emotion_steered_generation/{dataset_name}/steered_outputs.jsonl
输出 Output: outputs/{model_name}/03_emotion_steered_generation/{dataset_name}/labeled_results.jsonl
"""

import os, json, argparse, time, math, sys
from pathlib import Path
from typing import Dict, Any

# 6种情绪
# 6 emotions
EMOTIONS = ["anger","sadness","happiness","fear","surprise","disgust"]

# 3种极性
# 3 valences
VALENCES = ["positive", "neutral", "negative"]

# ========= OpenAI 客户端 =========
# OpenAI Client
from openai import OpenAI

# 从环境变量读取 API key
# Read API key from environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
if not OPENAI_API_KEY:
    print("[WARNING] OPENAI_API_KEY not found in environment variables. Please set it before running.")
    print("          export OPENAI_API_KEY='your-api-key-here'")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# GPT系统提示
# GPT system prompt
SYSTEM = f'''
    You are a careful rater. 
    Given a target emotion and a text, 
    decide if the text's STYLE matches the target emotion among:
    {EMOTIONS}
    Focus on tone/attitude, not content valence.
'''

# GPT用户提示模板
# GPT user prompt template
USER_TMPL = '''
    Target emotion: {emotion}
    Text:
    {text}
    Decide if the text's STYLE matches the target emotion.
    Return a compact JSON with keys exactly:
    {{
    "match": <0 or 1>, 
    "reason": <short string>
    }}
'''

def extract_json_from_response(response: str) -> str:
    """
    从GPT响应中提取JSON内容，处理markdown格式
    Extract JSON content from GPT response, handle markdown format
    """
    response = response.strip()
    
    # 如果包含markdown代码块，提取其中的JSON
    # If contains markdown code block, extract JSON from it
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    elif "```" in response:
        # 处理没有json标识的代码块
        # Handle code block without json identifier
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1:
            return response[start:end].strip()
    
    # 如果没有代码块，直接返回原内容
    # If no code block, return original content
    return response

def ask_llm(emotion: str, text: str, model: str, max_retries=4, backoff=1.8) -> Dict[str, Any]:
    """
    调用GPT进行情绪匹配判断
    Call GPT to judge emotion matching
    """
    if client is None:
        return {"match": 0, "reason": "no-openai-client"}
    
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system", "content": SYSTEM},
                    {"role":"user", "content": USER_TMPL.format(emotion=emotion, text=text)}
                ],
            )
            out = resp.choices[0].message.content
            print(f"      [{emotion}] GPT: {out[:80]}{'...' if len(out) > 80 else ''}")
            
            # 提取并解析 JSON
            # Extract and parse JSON
            json_str = extract_json_from_response(out)
            j = json.loads(json_str)
            
            # 基础字段兜底
            # Fallback for basic fields
            if "match" not in j:
                j = {"match": 0, "reason": "invalid-json"}
            if "reason" not in j:
                j["reason"] = "no-reason-provided"
                
            return j
        except Exception as e:
            print(f"      [{emotion}] Error: {e}")
            if i == max_retries-1:
                return {"match": 0, "reason": f"error:{type(e).__name__}"}
            time.sleep(backoff ** i)

def get_processed_keys(output_file):
    """
    获取已处理的样本key集合
    Get processed sample keys
    """
    if not output_file.exists():
        return set()
    
    processed_keys = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 使用 skeleton_id + event_valence 作为唯一key
                # Use skeleton_id + event_valence as unique key
                key = f"{obj['skeleton_id']}___{obj.get('event_valence', 'neutral')}"
                processed_keys.add(key)
            except:
                continue
    return processed_keys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama32_3b",
                       help="模型文件夹名称 Model folder name")
    parser.add_argument("--dataset_name", type=str, default="test_set",
                       help="数据集名称 Dataset name")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini",
                       help="GPT模型名称 GPT model name")
    args = parser.parse_args()
    
    # 构建路径
    # Build paths
    model_name = args.model_name
    dataset_name = args.dataset_name
    
    input_jsonl = Path("outputs") / model_name / "03_emotion_steered_generation" / dataset_name / "steered_outputs.jsonl"
    output_dir = Path("outputs") / model_name / "03_emotion_steered_generation" / dataset_name
    output_jsonl = output_dir / "labeled_results.jsonl"
    
    if not input_jsonl.exists():
        print(f"[ERROR] Missing input file: {input_jsonl}")
        sys.exit(1)
    
    if client is None:
        print(f"[ERROR] OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    print("情绪引导生成文本的GPT打标")
    print("GPT Labeling for Emotion-Steered Generated Text")
    print("=" * 60)
    print(f"[INFO] Input: {input_jsonl}")
    print(f"[INFO] Output: {output_jsonl}")
    print(f"[INFO] GPT Model: {args.gpt_model}")
    
    # 检查已处理的样本
    # Check processed samples
    processed_keys = get_processed_keys(output_jsonl)
    if len(processed_keys) > 0:
        print(f"[INFO] Found {len(processed_keys)} already processed records. Will skip them.")
    
    total_pairs = 0
    total_matches = 0
    per_emotion_stats = {e: {"n": 0, "matches": 0} for e in EMOTIONS}
    per_valence_stats = {v: {"n": 0, "matches": 0} for v in VALENCES}
    processed_samples = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "a", encoding="utf-8") as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line: 
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                continue
            
            skeleton_id = item["skeleton_id"]
            theme = item.get("theme", "")
            scenario = item.get("scenario", "")
            event_valence = item.get("event_valence", "neutral")
            event = item.get("event", "")
            emotion_texts = item.get("emotion_texts", {})
            parameters = item.get("parameters", {})
            
            # 检查是否已处理
            # Check if already processed
            sample_key = f"{skeleton_id}___{event_valence}"
            if sample_key in processed_keys:
                print(f"[SKIP] Line {line_num}: {sample_key} already processed")
                continue
            
            print(f"\n[{line_num}] Processing {skeleton_id} - {event_valence} - {theme}")
            
            # 对每种情绪生成文本进行打标
            # Label each emotion's generated text
            emotion_labels = {}
            for emotion, text in emotion_texts.items():
                if emotion not in EMOTIONS:
                    continue
                    
                if not isinstance(text, str) or text.startswith("[ERROR]"):
                    # 处理错误情况
                    # Handle error cases
                    label = {"match": 0, "reason": "error-text"}
                else:
                    print(f"    [{emotion}] Text: {text[:80]}{'...' if len(text) > 80 else ''}")
                    label = ask_llm(emotion, text, args.gpt_model)
                
                emotion_labels[emotion] = label
                total_pairs += 1
                per_emotion_stats[emotion]["n"] += 1
                per_valence_stats[event_valence]["n"] += 1
                
                if bool(label.get("match", 0)):
                    total_matches += 1
                    per_emotion_stats[emotion]["matches"] += 1
                    per_valence_stats[event_valence]["matches"] += 1
                
                # 添加小延迟避免API限制
                # Add small delay to avoid API rate limits
                time.sleep(0.1)
            
            # 保存结果
            # Save result
            result = {
                "skeleton_id": skeleton_id,
                "theme": theme,
                "scenario": scenario,
                "event_valence": event_valence,
                "event": event,
                "emotion_texts": emotion_texts,
                "emotion_labels": emotion_labels,  # {emotion: {match, reason}}
                "parameters": parameters,
                "timestamp": int(time.time()),
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()  # 强制刷新缓冲区 Force flush buffer
            
            processed_samples += 1
            print(f"  [SAVED] {sample_key}")
            
            # 每处理10个样本显示进度
            # Show progress every 10 samples
            if processed_samples % 10 == 0:
                current_acc = total_matches / max(total_pairs, 1)
                print(f"[progress] {processed_samples} samples processed, {total_pairs} pairs labeled, accuracy: {current_acc:.3f}")

    # 计算最终统计
    # Calculate final statistics
    overall_accuracy = total_matches / max(total_pairs, 1)
    
    print(f"\n{'=' * 60}")
    print(f"情绪打标总结 Emotion Labeling Summary ({args.gpt_model})")
    print(f"{'=' * 60}")
    print(f"Total samples processed: {processed_samples}")
    print(f"Total emotion-text pairs: {total_pairs}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    
    print(f"\n按情绪统计 Per-emotion breakdown:")
    for emotion in EMOTIONS:
        n = per_emotion_stats[emotion]["n"]
        matches = per_emotion_stats[emotion]["matches"]
        acc = matches / n if n > 0 else 0.0
        print(f"  {emotion:9s}: n={n:4d} | matches={matches:4d} | acc={acc:.3f}")
    
    print(f"\n按极性统计 Per-valence breakdown:")
    for valence in VALENCES:
        n = per_valence_stats[valence]["n"]
        matches = per_valence_stats[valence]["matches"]
        acc = matches / n if n > 0 else 0.0
        print(f"  {valence:9s}: n={n:4d} | matches={matches:4d} | acc={acc:.3f}")
    
    print(f"\n[OK] All results saved to: {output_jsonl}")

if __name__ == "__main__":
    main()
