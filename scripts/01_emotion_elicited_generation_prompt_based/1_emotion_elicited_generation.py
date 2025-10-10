# scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py
# -*- coding: utf-8 -*-
"""
情绪引导文本生成脚本
Emotion-Elicited Text Generation Script

纯文本生成脚本：使用情绪引导 prompt 生成文本
Pure text generation script: Uses emotion-guided prompts to generate text

- 输入 Input: data/sev.jsonl 或 data/test_set.jsonl 
- 输出 Output: outputs/{model_name}/01_emotion_elicited_generation_prompt_based/generated/{dataset_name}_generated.jsonl
"""

import os, json, time, argparse
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# HuggingFace Token
HF_TOKEN = 'Your HuggingFace Token'
login(token=HF_TOKEN)

# 情绪类型列表 Emotion types
EMOTIONS = ["anger","sadness","happiness","fear","surprise","disgust"]
# 事件类型列表 Event types
VALENCES = ["positive","neutral","negative"]


def build_messages_for_emotion(emotion: str, scenario: str, event: str):
    """
    构建情绪引导的对话消息
    Build emotion-guided conversation messages
    """
    system = f"""
Always reply in {emotion}.
Keep the reply to at most two sentences.
""".strip()
    user = f"{scenario}\n{event}"
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def iter_user_inputs(path: Path):
    """
    迭代读取输入文件
    Iterate through input file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if "scenario" in obj and "event" in obj:
                yield obj




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None,
                       help="输入数据路径 Input data path，如 e.g. data/sev.jsonl 或 data/test_set.jsonl")
    parser.add_argument("--both", action="store_true",
                       help="处理sev和test_set两个数据集 Process both sev and test_set datasets")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="模型名称 Model name")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备 Device")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32","bfloat16","float16"],
                       help="数据类型 Data type")
    parser.add_argument("--attn_impl", type=str, default="eager",
                       help="注意力实现 Attention implementation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="最大生成token数 Max new tokens to generate")
    parser.add_argument("--seed", type=int, default=1234,
                       help="随机种子 Random seed")
    parser.add_argument("--valences", type=str, default="positive,neutral,negative",
                       help="效价列表 Valence list")
    parser.add_argument("--emotions", type=str, default="anger,sadness,happiness,fear,surprise,disgust",
                       help="情绪列表 Emotion list")
    parser.add_argument("--skip_if_exists", action="store_true", default=True, 
                       help="跳过已处理的项目 Skip already processed items")
    parser.add_argument("--no_skip", action="store_true", 
                       help="重新处理所有项目 Reprocess all items")
    args = parser.parse_args()
    
    # 确定输入文件列表
    # Determine input file list
    if args.both:
        input_paths = [Path("data/sev.jsonl"), Path("data/test_set.jsonl")]
    elif args.input_path:
        input_paths = [Path(args.input_path)]
    else:
        parser.error("必须指定 --input_path 或 --both | Must specify --input_path or --both")
        return
    
    # 从模型名称生成简化的文件夹名
    # Generate simplified folder name from model name
    # meta-llama/Llama-3.2-3B-Instruct -> llama32_3b
    model_name = args.model.split('/')[-1].lower().replace('-', '').replace('.', '')
    if 'llama32' in model_name and '3b' in model_name:
        model_folder = 'llama32_3b'
    else:
        model_folder = model_name  # 其他模型使用原名 Use original name for other models

    # 数据类型映射
    # Data type mapping
    dmap = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dmap[args.dtype]
    torch.manual_seed(args.seed)

    # 加载分词器和模型（只加载一次）
    # Load tokenizer and model (only once)
    print("Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=HF_TOKEN)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        attn_implementation=args.attn_impl,
        token=HF_TOKEN
    )
    model.eval()
    print(f"Model loaded on device: {model.device}\n")

    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    valences = [v.strip() for v in args.valences.split(",") if v.strip()]
    
    # 处理每个输入文件
    # Process each input file
    for data_path in input_paths:
        if not data_path.exists():
            print(f"[WARNING] Input file not found: {data_path}, skipping...")
            continue
        
        dataset_name = data_path.stem  # sev 或 test_set
        
        # 构建输出路径
        # Build output path: outputs/{model_folder}/01_emotion_elicited_generation_prompt_based/generated/{dataset_name}_generated.jsonl
        out_dir = Path("outputs") / model_folder / "01_emotion_elicited_generation_prompt_based" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        jsonl_path = out_dir / f"{dataset_name}_generated.jsonl"
        
        # 加载已存在的 keys（用于断点续跑）
        # Load existing keys (for resuming from checkpoint)
        existing_keys = set()
        if args.skip_if_exists and not args.no_skip and jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        if "key" in obj:
                            existing_keys.add(obj["key"])
                    except:
                        continue

        print(f"{'='*70}")
        print(f"Processing dataset: {dataset_name}")
        print(f"Input: {data_path}")
        print(f"Output: {jsonl_path}")
        print(f"{'='*70}\n")

        total = 0
        skipped = 0
        start = time.time()

        with open(jsonl_path, "a", encoding="utf-8") as fout:
            for item in iter_user_inputs(data_path):
                theme    = item.get("theme", "")
                scenario = item["scenario"]
                events   = item["event"]          # dict: positive/neutral/negative 事件字典
                sid      = item.get("skeleton_id", "unknown")

                for val in valences:
                    if val not in events: continue
                    event = events[val]

                    for emo in emotions:
                        key = f"{sid}__{val}__{emo}".replace("/", "_")

                        # 断点续跑检查
                        # Checkpoint resuming check
                        if key in existing_keys:
                            skipped += 1
                            if skipped % 50 == 0:
                                print(f"[SKIP] {skipped} items skipped so far... (last: {key})")
                            continue

                        # 构造 messages 并生成
                        # Build messages and generate
                        msgs = build_messages_for_emotion(emo, scenario, event)
                        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        inputs = tok(prompt, return_tensors="pt").to(model.device)

                        # 贪婪生成
                        # Greedy generation
                        with torch.no_grad():
                            gen_ids = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                top_p=None,
                                top_k=None,
                                temperature=None,
                                eos_token_id=tok.eos_token_id,
                                pad_token_id=tok.pad_token_id,
                                use_cache=True,
                            )
                        out_ids = gen_ids[0][inputs.input_ids.shape[1]:]
                        gen_text = tok.decode(out_ids, skip_special_tokens=True).strip()

                        # 保存到 jsonl
                        # Save to jsonl
                        row = {
                            "key": key,
                            "skeleton_id": sid,
                            "theme": theme,
                            "valence": val,
                            "emotion": emo,
                            "scenario": scenario,
                            "event": event,
                            "gen_text": gen_text,
                            "meta": {
                                "model_id": args.model,
                                "dtype": args.dtype,
                                "device": args.device,
                                "attn_impl": args.attn_impl,
                                "max_new_tokens": args.max_new_tokens,
                                "seed": args.seed,
                                "input_len": int(inputs.input_ids.shape[1]),
                                "time": int(time.time()),
                            }
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fout.flush()  # 立即写入磁盘 Flush to disk immediately
                        
                        total += 1
                        if total % 20 == 0:
                            el = time.time() - start
                            rate = total / el if el > 0 else 0
                            print(f"[progress] generated={total} | last={key} | {el:.1f}s elapsed | {rate:.2f} items/s")

        elapsed = time.time() - start
        print(f"\n[OK] {dataset_name} completed. Generated {total} items, skipped {skipped} items.")
        print(f"     Time: {elapsed:.1f}s | Rate: {total/elapsed:.2f} items/s")
        print(f"     Output: {jsonl_path}\n")


if __name__ == "__main__":
    main()