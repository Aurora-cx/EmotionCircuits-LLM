#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/07_emotion_elicited_generation_circuit_based/3_baseline_text_generation.py
"""
基线文本生成脚本 - 不进行任何情绪注入，只生成原始回复
Baseline text generation script - no emotion injection, only generate original responses

功能 Functions:
  1. 加载用户输入测试数据
     Load user input test data
  2. 不进行任何情绪注入，只做纯文本生成
     No emotion injection, only pure text generation
  3. 对所有事件极性（positive、neutral、negative）都生成基线回复
     Generate baseline responses for all event valences (positive, neutral, negative)
  4. 用于对比情绪注入前后的文本差异
     Used to compare text differences before and after emotion injection

输入输出 Input/Output:
  - 输入 Input:  data/user_inputs_test_1.jsonl - 用户输入测试数据 / User input test data
  - 输出 Output: outputs/{model_name}/07_emotion_elicited_generation_circuit_based/baseline_generation/baseline_text_generation_outputs.jsonl
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============== 路径与基本配置 ==============
# ============== Paths and Basic Configuration ==============
# 项目根目录：自动获取脚本所在位置的上两级目录
# Project root: automatically get the directory two levels up from the script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "llama32_3b"
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

def build_messages(scenario: str, event: str):
    """
    构建对话消息
    Build conversation messages
    """
    system = 'Keep the reply to at most two sentences.'
    user = f"{scenario}\n{event}"
    return [{"role":"system","content":system},{"role":"user","content":user}]

def generate_text(model, tok, messages, max_new_tokens=100, do_sample=False):
    """
    生成文本
    Generate text
    
    参数 Args:
        model: 语言模型 language model
        tok: tokenizer
        messages: 对话消息 conversation messages
        max_new_tokens: 最大新token数 max new tokens
        do_sample: 是否采样 whether to sample
    
    返回 Returns: 生成的文本 generated text
    """
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
        min_new_tokens=10,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )
    out_ids = gen[0][inputs.input_ids.shape[1]:]
    return tok.decode(out_ids, skip_special_tokens=True).strip()

def load_user_inputs(data_path: str):
    """
    加载用户输入数据
    Load user input data
    """
    inputs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "event" in obj and "scenario" in obj:
                    inputs.append(obj)
            except json.JSONDecodeError:
                continue
    return inputs

# ============== 主函数 ==============
# ============== Main Function ==============
def main():
    """
    主函数
    Main function
    """
    ap = argparse.ArgumentParser(description="基线文本生成（无情绪注入）Baseline text generation (no emotion injection)")
    ap.add_argument("--model_name", type=str, default=MODEL_NAME,
                    help="模型名称 Model name")
    ap.add_argument("--max_new_tokens", type=int, default=100,
                    help="最大新token数 Max new tokens")
    ap.add_argument("--data_path", type=str, default="data/user_inputs_test_1.jsonl",
                    help="用户输入数据路径 User input data path")
    ap.add_argument("--output_filename", type=str, default="baseline_text_generation_outputs.jsonl",
                    help="输出文件名 Output filename")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="设备 Device")
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print(f"基线文本生成（无情绪注入）Baseline Text Generation (No Emotion Injection)")
    print(f"{'='*70}")
    print(f"[配置 CONFIG]")
    print(f"  - do_sample: False (贪婪采样 greedy sampling)")
    print(f"  - repetition_penalty: 1.1")
    print(f"  - max_new_tokens: {args.max_new_tokens}")
    print(f"  - 事件极性 Event valences: {VALENCES}")
    print(f"  - 注入状态 Injection status: 无注入（基线）No injection (baseline)")
    print(f"  - 设备 Device: {args.device}")
    print()

    # 1) 加载用户输入数据
    # 1) Load user input data
    data_path = PROJECT_ROOT / args.data_path
    user_inputs = load_user_inputs(str(data_path))
    print(f"[+] 已加载 Loaded {len(user_inputs)} user inputs from {data_path}")

    # 2) 加载模型
    # 2) Load model
    print(f"\n[+] 正在加载模型 Loading model: {MODEL}...")
    
    # HuggingFace token处理
    # HuggingFace token handling
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map=args.device,
        token=HF_TOKEN if HF_TOKEN else True
    )
    print(f"[+] 模型已加载 Model loaded successfully!")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN if HF_TOKEN else True)
    print(f"[+] Tokenizer已加载 Tokenizer loaded successfully!")
    print()

    # 3) 处理每个用户输入
    # 3) Process each user input
    start_time = time.time()
    output_dir = PROJECT_ROOT / "outputs" / args.model_name / "07_emotion_elicited_generation_circuit_based" / "baseline_generation"
    ensure_dir(output_dir)
    output_path = output_dir / args.output_filename

    print(f"[+] 开始生成基线文本 Starting baseline text generation...")
    print(f"[+] 输出路径 Output path: {output_path}\n")

    for i, user_input in enumerate(user_inputs):
        skeleton_id = user_input.get("skeleton_id", f"input_{i}")
        theme = user_input.get("theme", "Unknown")
        scenario = user_input["scenario"]
        
        print(f"[{i+1}/{len(user_inputs)}] 处理中 Processing {skeleton_id} - {theme}")
        
        # 对每种事件极性生成基线文本
        # Generate baseline text for each event valence
        valence_results = {}
        for valence in VALENCES:
            if valence not in user_input["event"]:
                print(f"  [!] 警告 WARNING: {valence} event not found in {skeleton_id}")
                continue
                
            event_text = user_input["event"][valence]
            print(f"  处理 Processing {valence} event...")
            
            try:
                # 生成基线文本（无情绪注入）
                # Generate baseline text (no emotion injection)
                msgs = build_messages(scenario, event_text)
                text = generate_text(model, tokenizer, msgs, max_new_tokens=args.max_new_tokens, do_sample=False)
                valence_results[valence] = text
                
                print(f"    [{valence}] {text}")
                
            except Exception as e:
                valence_results[valence] = f"[ERROR] {str(e)}"
                print(f"    [{valence}] 错误 ERROR: {str(e)}")
        
        # 构建结果
        # Build result
        result = {
            "skeleton_id": skeleton_id,
            "theme": theme,
            "scenario": scenario,
            "valence_results": valence_results,
            "parameters": {
                "do_sample": False,
                "repetition_penalty": 1.1,
                "max_new_tokens": args.max_new_tokens,
                "no_repeat_ngram_size": 3,
                "min_new_tokens": 10,
                "injection_method": "baseline",
                "valences": VALENCES,
            },
            "timestamp": int(time.time()),
        }

        # 立即保存这个样本的结果
        # Save this sample's result immediately
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"  [✓] 已保存 SAVED: {skeleton_id} results saved to {output_path}")

        # 每处理10个样本显示进度
        # Show progress every 10 samples
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[进度 Progress] {i+1}/{len(user_inputs)} completed, 耗时 elapsed: {elapsed:.1f}s\n")
    
    total_time = time.time() - start_time
    total_generations = len(user_inputs) * len(VALENCES)
    
    print(f"\n{'='*70}")
    print(f"[完成] 全部完成！[Done] All done!")
    print(f"{'='*70}")
    print(f"  处理的输入 Processed inputs: {len(user_inputs)}")
    print(f"  每个输入的事件极性 Event valences per input: {len(VALENCES)}")
    print(f"  总生成数 Total generations: {total_generations}")
    print(f"  总耗时 Total time: {total_time:.1f}s")
    print(f"  平均每输入耗时 Average time per input: {total_time/len(user_inputs):.1f}s")
    print(f"  所有结果已保存到 All results saved to: {output_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

