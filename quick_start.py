#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# quick_start.py
"""
快速开始：情绪电路调控演示脚本
Quick Start: Emotion Circuit Steering Demo

这是一个开箱即用的演示脚本，使用已训练好的情绪电路对文本生成进行情绪调控
This is a ready-to-use demo script that uses pre-trained emotion circuits to steer text generation

使用方法 Usage:
    python quick_start.py --input_text "I just finished my project" --emotion happiness --scale 1.0
    
    python quick_start.py --input_text "我刚刚完成了我的项目" --emotion happiness --scale 1.0

支持的情绪 Supported Emotions:
    - anger (愤怒)
    - sadness (悲伤)
    - happiness (快乐)
    - fear (恐惧)
    - disgust (厌恶)
    - surprise (惊讶)

推荐系数 Recommended Scale:
    - anger: 0.8 (anger情绪建议使用0.8以获得最佳效果 Recommended 0.8 for best results)
    - 其他情绪 Other emotions: 1.0
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============== 配置 ==============
# ============== Configuration ==============
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "llama32_3b"
D_MODEL = 3072
HEADS_PER_LAYER = 24
HEAD_DIM = D_MODEL // HEADS_PER_LAYER

EMOTIONS = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]

# ============== 数据加载 ==============
# ============== Data Loading ==============
def load_emotion_circuit(emotion):
    """
    加载情绪电路配置
    Load emotion circuit configuration
    """
    circuit_file = PROJECT_ROOT / "outputs" / MODEL_NAME / "06_emotion_circuit_integration" / "global_circuit" / f"{emotion}.json"
    
    if not circuit_file.exists():
        raise FileNotFoundError(f"电路文件不存在 Circuit file not found: {circuit_file}")
    
    with open(circuit_file, 'r') as f:
        data = json.load(f)
    
    neuron_ids = {}
    head_ids = {}
    for layer_info in data['layers']:
        layer_idx = layer_info['layer']
        neuron_ids[layer_idx] = layer_info['mlp']['neurons']
        head_ids[layer_idx] = layer_info['attention']['heads']
    
    return neuron_ids, head_ids

def load_emotion_residuals(emotion):
    """
    加载情绪残差向量
    Load emotion residual vectors
    """
    # MLP残差
    # MLP residuals
    mlp_file = PROJECT_ROOT / "outputs" / MODEL_NAME / "05_emotion_diff_vector_computation" / "mlp_emotion_diff" / "emo_diff_all.npz"
    data = np.load(mlp_file)
    
    neuron_diffs = {}
    for layer in range(28):
        key = f"{emotion}/diff/L{layer}"
        if key in data:
            neuron_diffs[layer] = data[key].astype(np.float32)
        else:
            neuron_diffs[layer] = np.zeros(8192, dtype=np.float32)
    
    # Attention残差
    # Attention residuals
    head_diffs = {}
    attn_dir = PROJECT_ROOT / "outputs" / MODEL_NAME / "05_emotion_diff_vector_computation" / "attention_emotion_diff" / "emo_diff" / emotion
    
    for layer in range(28):
        layer_file = attn_dir / f"L{layer}.npy"
        if layer_file.exists():
            head_diffs[layer] = np.load(layer_file).astype(np.float32)
        else:
            head_diffs[layer] = np.zeros(D_MODEL, dtype=np.float32)
    
    return neuron_diffs, head_diffs

def create_head_mask(head_indices):
    """
    创建attention head掩码
    Create attention head mask
    """
    mask = np.zeros(D_MODEL, dtype=np.float32)
    for head_id in head_indices:
        start_idx = head_id * HEAD_DIM
        end_idx = start_idx + HEAD_DIM
        mask[start_idx:end_idx] = 1.0
    return torch.from_numpy(mask)

# ============== 情绪注入器 ==============
# ============== Emotion Injector ==============
class EmotionCircuitInjector:
    """
    情绪电路注入器
    Emotion circuit injector
    """
    def __init__(self, emotion, neuron_ids, head_ids, neuron_diffs, head_diffs, scale_factor=1.0, device="cpu"):
        self.emotion = emotion
        self.neuron_ids = neuron_ids
        self.head_ids = head_ids
        self.neuron_diffs = neuron_diffs
        self.head_diffs = head_diffs
        self.scale_factor = scale_factor
        self.device = device
        self.handles = []
    
    def _hook_mlp(self, layer_idx):
        def hook_fn(module, inputs, output):
            x = inputs[0]
            gate = module.gate_proj(x)
            up = module.up_proj(x)
            act = module.act_fn(gate)
            a = act * up
            
            if layer_idx in self.neuron_ids and layer_idx in self.neuron_diffs:
                neuron_indices = self.neuron_ids[layer_idx]
                if len(neuron_indices) > 0:
                    diff_vec = torch.from_numpy(self.neuron_diffs[layer_idx]).to(a.device, dtype=a.dtype)
                    idx_tensor = torch.tensor(neuron_indices, device=a.device, dtype=torch.long)
                    a[:, -1, idx_tensor] += self.scale_factor * diff_vec[idx_tensor]
            
            return module.down_proj(a)
        return hook_fn
    
    def _hook_o_proj(self, layer_idx):
        def hook_fn(module, inputs):
            if layer_idx in self.head_ids and layer_idx in self.head_diffs:
                head_indices = self.head_ids[layer_idx]
                if len(head_indices) > 0:
                    x = inputs[0]
                    head_diff = torch.from_numpy(self.head_diffs[layer_idx]).to(x.device, dtype=x.dtype)
                    mask = create_head_mask(head_indices).to(x.device, dtype=x.dtype)
                    masked_diff = head_diff * mask
                    x = x + self.scale_factor * masked_diff.unsqueeze(0).unsqueeze(0)
                    return (x,)
            return inputs
        return hook_fn
    
    def register(self, model):
        for layer_idx in range(28):
            if layer_idx in self.neuron_ids:
                mlp_layer = model.model.layers[layer_idx].mlp
                handle = mlp_layer.register_forward_hook(self._hook_mlp(layer_idx))
                self.handles.append(handle)
            
            if layer_idx in self.head_ids:
                attn_layer = model.model.layers[layer_idx].self_attn
                handle = attn_layer.o_proj.register_forward_pre_hook(self._hook_o_proj(layer_idx))
                self.handles.append(handle)
    
    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

# ============== 文本生成 ==============
# ============== Text Generation ==============
def generate_with_emotion(text, emotion, scale_factor, model, tokenizer, device):
    """
    生成情绪调控后的文本
    Generate emotion-steered text
    """
    # 加载情绪电路和残差
    # Load emotion circuit and residuals
    print(f"\n[1/4] 加载情绪电路 Loading {emotion} emotion circuit...")
    neuron_ids, head_ids = load_emotion_circuit(emotion)
    
    # 统计选中的组件
    # Count selected components
    total_neurons = sum(len(ids) for ids in neuron_ids.values())
    total_heads = sum(len(ids) for ids in head_ids.values())
    print(f"      ✓ 已选中 Selected: {total_neurons} neurons + {total_heads} attention heads")
    
    print(f"[2/4] 加载情绪残差 Loading {emotion} residual vectors...")
    neuron_diffs, head_diffs = load_emotion_residuals(emotion)
    print(f"      ✓ 已加载 Loaded: 28层MLP残差 + 28层Attention残差")
    
    # 创建注入器
    # Create injector
    print(f"[3/4] 注册情绪注入钩子 Registering emotion injection hooks...")
    injector = EmotionCircuitInjector(
        emotion, neuron_ids, head_ids, neuron_diffs, head_diffs,
        scale_factor=scale_factor, device=device
    )
    injector.register(model)
    print(f"      ✓ 已注册 Registered hooks for emotion steering")
    
    try:
        # 生成文本
        # Generate text
        print(f"[4/4] 生成情绪调控文本 Generating {emotion}-steered text...")
        messages = [
            {"role": "system", "content": "Keep the reply to at most two sentences."},
            {"role": "user", "content": text}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            min_new_tokens=10,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"      ✓ 生成完成 Generation completed")
        
        return generated_text
        
    finally:
        injector.remove()

def generate_baseline(text, model, tokenizer, device):
    """
    生成基线文本（无情绪注入）
    Generate baseline text (no emotion injection)
    """
    messages = [
        {"role": "system", "content": "Keep the reply to at most two sentences."},
        {"role": "user", "content": text}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        min_new_tokens=10,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generated_text

# ============== 主函数 ==============
# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(
        description="情绪电路调控快速演示 Emotion Circuit Steering Quick Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 Examples:
  python quick_start.py --input_text "I just got promoted at work!" --emotion happiness --scale 1.0
  python quick_start.py --input_text "The project deadline was moved up" --emotion fear --scale 1.0
  python quick_start.py --input_text "They ignored my suggestion again" --emotion anger --scale 0.8
        """
    )
    
    parser.add_argument("--input_text", type=str, required=True,
                       help="输入文本 Input text")
    parser.add_argument("--emotion", type=str, required=True, choices=EMOTIONS,
                       help=f"目标情绪 Target emotion: {', '.join(EMOTIONS)}")
    parser.add_argument("--scale", type=float, default=None,
                       help="注入系数 Injection scale (默认 default: anger=0.8, 其他 others=1.0)")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 Device (默认自动检测 default: auto-detect GPU/CPU)")
    parser.add_argument("--skip_baseline", action="store_true",
                       help="跳过基线生成，只输出情绪调控结果 Skip baseline generation")
    
    args = parser.parse_args()
    
    # 自动设置最佳scale
    # Auto set best scale
    if args.scale is None:
        args.scale = 0.8 if args.emotion == "anger" else 1.0
    
    # 自动检测设备
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
            print(f"✓ 检测到GPU，使用CUDA Detected GPU, using CUDA")
        else:
            args.device = "cpu"
            print(f"ℹ️  未检测到GPU，使用CPU No GPU detected, using CPU")
    
    print(f"\n{'='*70}")
    print(f"🎭 情绪电路调控演示 Emotion Circuit Steering Demo")
    print(f"{'='*70}")
    print(f"输入文本 Input: {args.input_text}")
    print(f"目标情绪 Emotion: {args.emotion}")
    print(f"注入系数 Scale: {args.scale}")
    print(f"设备 Device: {args.device}")
    print(f"{'='*70}\n")
    
    # 加载模型
    # Load model
    print(f"🔧 加载模型 Loading model: {MODEL}...")
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN if HF_TOKEN else True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map=args.device,
        token=HF_TOKEN if HF_TOKEN else True
    )
    print(f"   ✓ 模型加载完成 Model loaded\n")
    
    # 生成基线输出
    # Generate baseline output
    if not args.skip_baseline:
        print(f"{'='*70}")
        print(f"📝 基线输出 Baseline Output (无情绪调控 No emotion steering)")
        print(f"{'='*70}")
        baseline_text = generate_baseline(args.input_text, model, tokenizer, args.device)
        print(f"{baseline_text}")
        print()
    
    # 生成情绪调控输出
    # Generate emotion-steered output
    print(f"{'='*70}")
    print(f"🎭 情绪调控输出 {args.emotion.capitalize()}-Steered Output (系数 scale={args.scale})")
    print(f"{'='*70}")
    
    steered_text = generate_with_emotion(
        args.input_text, args.emotion, args.scale,
        model, tokenizer, args.device
    )
    
    print(f"\n{'='*70}")
    print(f"✨ 最终结果 Final Result:")
    print(f"{'='*70}")
    print(f"{steered_text}")
    print(f"{'='*70}\n")
    
    # 保存结果（可选）
    # Save result (optional)
    result = {
        "input": args.input_text,
        "emotion": args.emotion,
        "scale_factor": args.scale,
        "baseline": baseline_text if not args.skip_baseline else None,
        "steered": steered_text,
        "model": MODEL
    }
    
    output_dir = PROJECT_ROOT / "quick_start_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 使用时间戳避免覆盖
    # Use timestamp to avoid overwrite
    import time
    timestamp = int(time.time())
    output_file = output_dir / f"demo_{args.emotion}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 结果已保存到 Result saved to: {output_file}\n")

if __name__ == "__main__":
    main()

