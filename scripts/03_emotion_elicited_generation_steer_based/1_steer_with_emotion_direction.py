# scripts/03_emotion_elicited_generation_steer_based/1_steer_with_emotion_direction.py
# -*- coding: utf-8 -*-
"""
情绪引导生成脚本 - 使用情绪向量引导生成
Emotion-Steered Generation Script - Using Emotion Vectors to Guide Generation

使用提取的情绪方向向量对输入数据进行情绪引导生成
Use extracted emotion direction vectors to guide generation for input data

默认参数 Default parameters：
- layers: 11-20（作用层范围 Layers to apply steering）
- alpha: 8（引导强度 Steering strength）
- last_k: 1（作用位置数 Number of positions to steer）
- scale: rms（缩放模式 Scaling mode）
- dtype: float32（数据类型 Data type）
- direction_type: mlp（方向类型 Direction type: mlp or attention）

输入 Input: data/{dataset_name}.jsonl
输出 Output: outputs/{model_name}/03_emotion_steered_generation/{dataset_name}/steered_outputs.jsonl
"""

import os, json, argparse, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# HF token (优先使用环境变量，如果没有则尝试默认登录)
# HF token (prioritize env var, otherwise use default login)
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if HF_TOKEN:
    login(token=HF_TOKEN)

# 6种情绪
# 6 emotions
EMOS6 = ["anger","sadness","happiness","fear","surprise","disgust"]

# 3种极性（如果数据有多极性）
# 3 valences (if data has multiple valences)
VALENCES = ["positive", "neutral", "negative"]

def build_messages(scenario: str, event: str):
    """
    构建对话消息
    Build conversation messages
    """
    system = '''
    Keep the reply to at most two sentences.
    '''
    user = f"{scenario}\n{event}"
    return [{"role":"system","content":system},{"role":"user","content":user}]

def load_directions(path: Path):
    """
    加载情绪方向向量
    Load emotion direction vectors
    """
    obj = torch.load(path, map_location="cpu")
    dirs = obj["dirs"]; L = obj["layers"]; H = obj["hidden"]; emos = obj["emotions"]
    for e in dirs:
        if not isinstance(dirs[e], np.ndarray):
            dirs[e] = np.array(dirs[e], dtype=np.float32)
        else:
            dirs[e] = dirs[e].astype(np.float32)
    return dirs, L, H, emos

def parse_layers(arg: str):
    """
    解析层范围参数
    Parse layer range parameter
    """
    if "-" in arg:
        a,b = arg.split("-"); return list(range(int(a), int(b)+1))
    return [int(x) for x in arg.split(",") if x.strip()]

class EmotionSteerer:
    """
    情绪引导器
    Emotion Steerer
    """
    def __init__(self, model, dirs_np, target_emotion, layer_ids, alpha=8.0, last_k=1, scale_mode="rms"):
        """
        初始化情绪引导器
        Initialize emotion steerer
        """
        self.model = model
        self.alpha = float(alpha)
        self.layer_ids = list(layer_ids)
        self.last_k = int(last_k)
        self.scale_mode = scale_mode
        self.v = {l: torch.from_numpy(dirs_np[target_emotion][l]).to(model.device) for l in self.layer_ids}
        self.handles = []
        for l in self.layer_ids:
            h = self.model.model.layers[l].register_forward_hook(self._make_hook(l))
            self.handles.append(h)

    def _make_hook(self, layer_id: int):
        """
        创建hook函数
        Create hook function
        """
        v = self.v[layer_id]
        alpha = self.alpha
        last_k = self.last_k
        scale_mode = self.scale_mode
        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                hs = output[0].clone()  # 克隆以避免修改原始数据 Clone to avoid modifying original
                B, T, H = hs.shape
                # 对最后 K 个位置加偏移
                # Add offset to last K positions
                start = max(0, T - last_k)
                if scale_mode == "rms":
                    # 每个位置自适应缩放（不改方向）
                    # Adaptive scaling per position (preserve direction)
                    seg = hs[:, start:T, :]                              # [B,K,H]
                    rms = torch.sqrt((seg**2).mean(dim=-1, keepdim=True) + 1e-12)  # [B,K,1]
                    delta = alpha * v.view(1,1,H) * rms                  # [B,K,H]
                else:
                    delta = alpha * v.view(1,1,H)
                hs[:, start:T, :] = hs[:, start:T, :] + delta
                # 返回修改后的 tuple，保持原有格式
                # Return modified tuple, keep original format
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                B, T, H = hs.shape
                start = max(0, T - last_k)
                if scale_mode == "rms":
                    seg = hs[:, start:T, :]
                    rms = torch.sqrt((seg**2).mean(dim=-1, keepdim=True) + 1e-12)
                    delta = alpha * v.view(1,1,H) * rms
                else:
                    delta = alpha * v.view(1,1,H)
                hs[:, start:T, :] = hs[:, start:T, :] + delta
                return hs
        return hook

    def remove(self):
        """
        移除所有hooks
        Remove all hooks
        """
        for h in self.handles: h.remove()
        self.handles.clear()

@torch.no_grad()
def generate_text(model, tok, messages, max_new_tokens=500):
    """
    生成文本 - 使用贪婪采样
    Generate text - using greedy sampling
    """
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 贪婪采样 Greedy sampling
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        use_cache=True,
        # 防止截断的参数 Parameters to prevent truncation
        min_new_tokens=10,  # 最少生成10个token Min 10 tokens
        repetition_penalty=1.05,  # 轻微惩罚重复，避免循环 Slight repetition penalty
        no_repeat_ngram_size=3,  # 避免3-gram重复 Avoid 3-gram repetition
    )
    out_ids = gen[0][inputs.input_ids.shape[1]:]
    return tok.decode(out_ids, skip_special_tokens=True).strip()

def load_user_inputs(data_path: str, has_valence: bool = False):
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
                if has_valence:
                    # 检查是否包含所有三种极性的事件
                    # Check if contains all three valences
                    if "event" in obj and "scenario" in obj:
                        event_data = obj["event"]
                        if all(valence in event_data for valence in VALENCES):
                            inputs.append(obj)
                else:
                    # 简单格式：只需要scenario和event
                    # Simple format: only need scenario and event
                    if "event" in obj and "scenario" in obj:
                        inputs.append(obj)
            except json.JSONDecodeError:
                continue
    return inputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="llama32_3b",
                   help="模型文件夹名称 Model folder name")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HuggingFace模型名称 HuggingFace model name")
    ap.add_argument("--dataset_name", type=str, default="test_set",
                   help="数据集名称 Dataset name")
    ap.add_argument("--direction_type", type=str, default="mlp", choices=["mlp", "attention"],
                   help="方向类型 Direction type")
    ap.add_argument("--layers", type=str, default="11-20", 
                   help="层范围 Layer range")
    ap.add_argument("--alpha", type=float, default=8.0, 
                   help="引导强度 Steering strength alpha")
    ap.add_argument("--last_k", type=int, default=1, 
                   help="作用位置数 Number of positions to steer")
    ap.add_argument("--scale", type=str, default="rms", choices=["unit","rms"], 
                   help="缩放模式 Scaling mode")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float16","bfloat16","float32"], 
                   help="数据类型 Data type")
    ap.add_argument("--max_new_tokens", type=int, default=100,
                   help="最大生成token数 Max new tokens")
    ap.add_argument("--data_path", type=str, default=None,
                   help="输入数据路径 Input data path (如不指定则使用默认路径)")
    ap.add_argument("--has_valence", action="store_true",
                   help="数据是否包含多极性 Whether data has multiple valences")
    ap.add_argument("--device", type=str, default="cuda:0",
                   help="设备 Device")
    args = ap.parse_args()

    print("情绪引导生成脚本")
    print("Emotion-Steered Generation Script")
    print("=" * 60)
    print(f"参数设置 Parameters:")
    print(f"  model_name: {args.model_name}")
    print(f"  dataset_name: {args.dataset_name}")
    print(f"  direction_type: {args.direction_type}")
    print(f"  layers: {args.layers}")
    print(f"  alpha: {args.alpha}")
    print(f"  last_k: {args.last_k}")
    print(f"  scale: {args.scale}")
    print(f"  dtype: {args.dtype}")
    print(f"  采样方式 Sampling: 贪婪采样 Greedy (do_sample=False)")

    # 构建路径
    # Build paths
    model_name = args.model_name
    dataset_name = args.dataset_name
    
    # 情绪方向向量路径
    # Emotion direction vectors path
    directions_file = Path("outputs") / model_name / "02_emotion_directions" / f"emo_directions_{args.direction_type}.pt"
    
    # 输入数据路径
    # Input data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = Path("data") / f"{dataset_name}.jsonl"
    
    # 输出路径
    # Output path
    out_dir = Path("outputs") / model_name / "03_emotion_steered_generation" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "steered_outputs.jsonl"

    # 1) 加载情绪方向向量
    # Load emotion direction vectors
    if not directions_file.exists():
        print(f"[ERROR] Direction file not found: {directions_file}")
        return
    dirs, L, Hdim, emos = load_directions(directions_file)
    assert set(EMOS6).issubset(set(emos))
    print(f"[dirs] loaded from {directions_file}: layers={L}, H={Hdim}")

    # 2) 加载用户输入数据
    # Load user input data
    if not Path(data_path).exists():
        print(f"[ERROR] Data file not found: {data_path}")
        return
    user_inputs = load_user_inputs(data_path, has_valence=args.has_valence)
    print(f"[data] loaded {len(user_inputs)} user inputs from {data_path}")

    # 3) 加载模型
    # Load model
    torch_dtype = {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=HF_TOKEN if HF_TOKEN else True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map=args.device, token=HF_TOKEN if HF_TOKEN else True)
    model.eval()
    print(f"[model] loaded on {model.device}")

    # 4) 处理每个用户输入
    # Process each user input
    layer_ids = parse_layers(args.layers)
    start_time = time.time()
    processed = 0

    for i, user_input in enumerate(user_inputs):
        skeleton_id = user_input.get("skeleton_id", f"input_{i}")
        theme = user_input.get("theme", "Unknown")
        scenario = user_input["scenario"]
        
        print(f"\n[{i+1}/{len(user_inputs)}] Processing {skeleton_id} - {theme}")
        
        if args.has_valence:
            # 对每种极性和每种情绪生成文本
            # Generate text for each valence and emotion
            for valence in VALENCES:
                event_text = user_input["event"][valence]
                print(f"  Processing {valence} valence...")
                
                emotion_texts = {}
                for emo in EMOS6:
                    steerer = EmotionSteerer(
                        model, dirs, emo, layer_ids, 
                        alpha=args.alpha, last_k=args.last_k, scale_mode=args.scale
                    )
                    try:
                        msgs = build_messages(scenario, event_text)
                        text = generate_text(model, tok, msgs, max_new_tokens=args.max_new_tokens)
                        emotion_texts[emo] = text
                        print(f"    [{emo}] {text}")
                    except Exception as e:
                        emotion_texts[emo] = f"[ERROR] {str(e)}"
                        print(f"    [{emo}] ERROR: {str(e)}")
                    finally:
                        steerer.remove()
                
                # 构建结果 - 每个极性单独一条记录
                # Build result - one record per valence
                result = {
                    "skeleton_id": skeleton_id,
                    "theme": theme,
                    "scenario": scenario,
                    "event_valence": valence,
                    "event": event_text,
                    "emotion_texts": emotion_texts,
                    "parameters": {
                        "layers": layer_ids,
                        "alpha": args.alpha,
                        "last_k": args.last_k,
                        "scale": args.scale,
                        "dtype": args.dtype,
                        "direction_type": args.direction_type,
                        "max_new_tokens": args.max_new_tokens,
                        "sampling": "greedy",
                    },
                    "timestamp": int(time.time()),
                }
                
                # 立即保存这个极性的结果
                # Save this valence's result immediately
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                print(f"  [SAVED] {skeleton_id} - {valence} saved to {output_path}")
        else:
            # 简单格式：单个事件，对每种情绪生成
            # Simple format: single event, generate for each emotion
            event_text = user_input["event"]
            valence = user_input.get("valence", "neutral")  # 默认为neutral
            
            emotion_texts = {}
            for emo in EMOS6:
                steerer = EmotionSteerer(
                    model, dirs, emo, layer_ids, 
                    alpha=args.alpha, last_k=args.last_k, scale_mode=args.scale
                )
                try:
                    msgs = build_messages(scenario, event_text)
                    text = generate_text(model, tok, msgs, max_new_tokens=args.max_new_tokens)
                    emotion_texts[emo] = text
                    print(f"  [{emo}] {text}")
                except Exception as e:
                    emotion_texts[emo] = f"[ERROR] {str(e)}"
                    print(f"  [{emo}] ERROR: {str(e)}")
                finally:
                    steerer.remove()
            
            # 构建结果 - 与样例文件格式一致
            # Build result - consistent with example file format
            result = {
                "skeleton_id": skeleton_id,
                "theme": theme,
                "scenario": scenario,
                "event_valence": valence,
                "event": event_text,
                "emotion_texts": emotion_texts,
                "parameters": {
                    "layers": layer_ids,
                    "alpha": args.alpha,
                    "last_k": args.last_k,
                    "scale": args.scale,
                    "dtype": args.dtype,
                    "direction_type": args.direction_type,
                    "max_new_tokens": args.max_new_tokens,
                    "sampling": "greedy",
                },
                "timestamp": int(time.time()),
            }
            
            # 立即保存这个样本的结果
            # Save this sample's result immediately
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            print(f"  [SAVED] {skeleton_id} - {valence} saved to {output_path}")
        
        processed += 1

        # 每处理10个样本显示进度
        # Show progress every 10 samples
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[progress] {i+1}/{len(user_inputs)} completed, elapsed: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n[OK] All done!")
    print(f"  Processed: {processed} user inputs")
    if args.has_valence:
        print(f"  Generated: {processed * len(VALENCES)} records ({processed} inputs × {len(VALENCES)} valences)")
        print(f"  Total emotion texts: {processed * len(VALENCES) * len(EMOS6)}")
    else:
        print(f"  Generated: {processed} records")
        print(f"  Total emotion texts: {processed * len(EMOS6)}")
    print(f"  Total time: {total_time:.1f}s")
    if processed > 0:
        print(f"  Average time per input: {total_time/processed:.1f}s")
    print(f"  All results saved to: {output_path}")

if __name__ == "__main__":
    main()
