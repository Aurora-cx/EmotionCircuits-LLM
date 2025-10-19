#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/07_emotion_elicited_generation_circuit_based/4_circuit_steer_all_valences.py
"""
全局电路情绪注入文本生成 - 所有事件极性
Global circuit emotion injection text generation - all event valences

功能 Functions:
  1. 基于全局电路分配结果进行情绪注入
     Perform emotion injection based on global circuit allocation results
  2. 对所有事件极性（positive、neutral、negative）都进行情绪注入测试
     Test emotion injection on all event valences (positive, neutral, negative)
  3. 生成情绪增强的文本回复
     Generate emotion-enhanced text responses
  4. 用于对比基线和情绪注入后的文本差异
     Used to compare text differences between baseline and emotion-injected responses

关键参数 Key Parameters:
  - anger 使用 scale_factor=0.8（最佳参数）
    anger uses scale_factor=0.8 (best parameter)
  - 其他情绪使用 scale_factor=1.0
    other emotions use scale_factor=1.0
  - 贪婪采样（do_sample=False）
    greedy sampling (do_sample=False)

输入输出 Input/Output:
  - 输入 Input:
    - data/user_inputs_test_1.jsonl - 用户输入测试数据 / User input test data
    - outputs/{model_name}/06_emotion_circuit_integration/global_circuit/{emotion}.json - 全局电路分配 / Global circuit allocation
    - outputs/{model_name}/05_emotion_diff_vector_computation/mlp_emotion_diff/emo_diff_all.npz - MLP差分向量 / MLP diff vectors
    - outputs/{model_name}/05_emotion_diff_vector_computation/attention_emotion_diff/emo_diff/{emotion}/L{layer}.npy - Attention差分向量 / Attention diff vectors
  - 输出 Output:
    - outputs/{model_name}/07_emotion_elicited_generation_circuit_based/circuit_steered_generation/circuit_steer_all_valences_outputs.jsonl
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
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
EMOS6 = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]
VALENCES = ["positive", "neutral", "negative"]

# ============== 数据加载 ==============
# ============== Data Loading ==============
def load_global_circuit_allocation(emotion, model_name=MODEL_NAME):
    """
    读取全局电路分配结果
    Read global circuit allocation results
    """
    json_path = PROJECT_ROOT / "outputs" / model_name / "06_emotion_circuit_integration" / "global_circuit" / f"{emotion}.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    neuron_ids = {}
    head_ids = {}
    for layer_info in data['layers']:
        layer_idx = layer_info['layer']
        neuron_ids[layer_idx] = layer_info['mlp']['neurons']
        head_ids[layer_idx] = layer_info['attention']['heads']
    
    return neuron_ids, head_ids

def load_emotion_diff_vectors(model_name, target_emotion):
    """
    加载MLP情绪差分向量Δμ
    Load MLP emotion difference vectors Δμ
    """
    mlp_diff_dir = PROJECT_ROOT / "outputs" / model_name / "05_emotion_diff_vector_computation" / "mlp_emotion_diff"
    npz_file = mlp_diff_dir / "emo_diff_all.npz"
    
    data = np.load(npz_file)
    diff_vectors = {}
    for layer in range(28):
        key = f"{target_emotion}/diff/L{layer}"
        if key in data:
            diff_vectors[layer] = data[key].astype(np.float32)
        else:
            diff_vectors[layer] = np.zeros(8192, dtype=np.float32)
    
    return diff_vectors

def load_head_differences(emotion, model_name=MODEL_NAME):
    """
    加载Attention head差分向量
    Load Attention head difference vectors
    """
    head_diff_dir = PROJECT_ROOT / "outputs" / model_name / "05_emotion_diff_vector_computation" / "attention_emotion_diff" / "emo_diff" / emotion
    head_diffs = {}
    
    for layer in range(28):
        layer_file = head_diff_dir / f"L{layer}.npy"
        if layer_file.exists():
            head_diffs[layer] = np.load(layer_file).astype(np.float32)
        else:
            head_diffs[layer] = np.zeros(3072, dtype=np.float32)
    
    return head_diffs

def create_head_mask(head_indices):
    """
    创建head mask
    Create head mask
    """
    D_MODEL = 3072
    HEADS_PER_LAYER = 24
    HEAD_DIM = D_MODEL // HEADS_PER_LAYER
    
    mask = np.zeros(D_MODEL, dtype=np.float32)
    for head_id in head_indices:
        start_idx = head_id * HEAD_DIM
        end_idx = start_idx + HEAD_DIM
        mask[start_idx:end_idx] = 1.0
        
    return torch.from_numpy(mask)

# ============== 情绪注入器 ==============
# ============== Emotion Injector ==============
class SimpleEmotionInjector:
    """
    简化的情绪注入器（用于文本生成）
    Simplified emotion injector (for text generation)
    """
    def __init__(self, emotion, neuron_ids, head_ids, neuron_diffs, head_diffs, scale_factor=1.0, device="cpu"):
        """
        初始化注入器
        Initialize injector
        """
        self.emotion = emotion
        self.neuron_ids = neuron_ids
        self.head_ids = head_ids
        self.neuron_diffs = neuron_diffs
        self.head_diffs = head_diffs
        self.scale_factor = scale_factor
        self.device = device
        self.handles = []
    
    def _hook_mlp(self, layer_idx):
        """
        MLP层的hook函数
        Hook function for MLP layer
        """
        def hook_fn(module, inputs, output):
            x = inputs[0]
            B, T, H = x.shape
            
            gate = module.gate_proj(x)
            up = module.up_proj(x)
            act = module.act_fn(gate)
            a = act * up
            
            # 注入选中的神经元残差
            # Inject residuals to selected neurons
            if layer_idx in self.neuron_ids and layer_idx in self.neuron_diffs:
                neuron_indices = self.neuron_ids[layer_idx]
                if len(neuron_indices) > 0:
                    diff_vec = torch.from_numpy(self.neuron_diffs[layer_idx]).to(a.device, dtype=a.dtype)
                    idx_tensor = torch.tensor(neuron_indices, device=a.device, dtype=torch.long)
                    a[:, -1, idx_tensor] += self.scale_factor * diff_vec[idx_tensor]
            
            out = module.down_proj(a)
            return out
        return hook_fn
    
    def _hook_o_proj(self, layer_idx):
        """
        o_proj的pre_forward_hook函数
        pre_forward_hook function for o_proj
        """
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
        """
        注册所有hook
        Register all hooks
        """
        for layer_idx in range(28):
            if layer_idx in self.neuron_ids:
                mlp_layer = model.model.layers[layer_idx].mlp
                handle = mlp_layer.register_forward_hook(self._hook_mlp(layer_idx))
                self.handles.append(handle)
            
            if layer_idx in self.head_ids:
                attn_layer = model.model.layers[layer_idx].self_attn
                o_proj_handle = attn_layer.o_proj.register_forward_pre_hook(self._hook_o_proj(layer_idx))
                self.handles.append(o_proj_handle)
    
    def remove(self):
        """
        移除所有hook
        Remove all hooks
        """
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

# ============== 文本生成 ==============
# ============== Text Generation ==============
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
    加载用户输入数据 - 包含所有事件极性
    Load user input data - includes all event valences
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
    ap = argparse.ArgumentParser(description="全局电路情绪注入文本生成 Global circuit emotion injection text generation")
    ap.add_argument("--model_name", type=str, default=MODEL_NAME,
                    help="模型名称 Model name")
    ap.add_argument("--scale_factor", type=float, default=1.0,
                    help="情绪注入强度（anger会自动使用0.8）Emotion injection strength (anger auto uses 0.8)")
    ap.add_argument("--max_new_tokens", type=int, default=100,
                    help="最大新token数 Max new tokens")
    ap.add_argument("--data_path", type=str, default="data/user_inputs_test_1.jsonl",
                    help="用户输入数据路径 User input data path")
    ap.add_argument("--output_filename", type=str, default="circuit_steer_all_valences_outputs.jsonl",
                    help="输出文件名 Output filename")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="设备 Device")
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print(f"全局电路情绪注入文本生成 Global Circuit Emotion Injection Text Generation")
    print(f"{'='*70}")
    print(f"[配置 CONFIG]")
    print(f"  - scale_factor: anger=0.8, 其他情绪 other emotions={args.scale_factor}")
    print(f"  - do_sample: False (贪婪采样 greedy sampling)")
    print(f"  - repetition_penalty: 1.1")
    print(f"  - max_new_tokens: {args.max_new_tokens}")
    print(f"  - 事件极性 Event valences: {VALENCES}")
    print(f"  - 情绪类型 Emotion types: {EMOS6}")
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
    output_dir = PROJECT_ROOT / "outputs" / args.model_name / "07_emotion_elicited_generation_circuit_based" / "circuit_steered_generation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / args.output_filename

    print(f"[+] 开始生成情绪注入文本 Starting emotion-injected text generation...")
    print(f"[+] 输出路径 Output path: {output_path}\n")

    for i, user_input in enumerate(user_inputs):
        skeleton_id = user_input.get("skeleton_id", f"input_{i}")
        theme = user_input.get("theme", "Unknown")
        scenario = user_input["scenario"]
        
        print(f"[{i+1}/{len(user_inputs)}] 处理中 Processing {skeleton_id} - {theme}")
        
        # 对每种事件极性和每种情绪生成文本
        # Generate text for each event valence and each emotion
        valence_results = {}
        for valence in VALENCES:
            if valence not in user_input["event"]:
                print(f"  [!] 警告 WARNING: {valence} event not found in {skeleton_id}")
                continue
                
            event_text = user_input["event"][valence]
            print(f"  处理 Processing {valence} event...")
            
            emotion_results = {}
            for emo in EMOS6:
                try:
                    # 加载该情绪的数据
                    # Load data for this emotion
                    neuron_ids, head_ids = load_global_circuit_allocation(emo, args.model_name)
                    neuron_diffs = load_emotion_diff_vectors(args.model_name, emo)
                    head_diffs = load_head_differences(emo, args.model_name)
                    
                    # 创建情绪注入器（使用最佳参数，anger使用0.8的scale）
                    # Create emotion injector (use best params, anger uses scale 0.8)
                    scale_factor = 0.8 if emo == "anger" else args.scale_factor
                    injector = SimpleEmotionInjector(
                        emo, neuron_ids, head_ids, neuron_diffs, head_diffs, 
                        scale_factor=scale_factor, device=args.device
                    )
                    
                    # 注册hooks
                    # Register hooks
                    injector.register(model)
                    
                    # 生成文本（使用最佳参数）
                    # Generate text (use best params)
                    msgs = build_messages(scenario, event_text)
                    text = generate_text(model, tokenizer, msgs, max_new_tokens=args.max_new_tokens, do_sample=False)
                    emotion_results[emo] = text
                    
                    # 清理hooks
                    # Clean up hooks
                    injector.remove()
                    
                    print(f"    [{emo}] {text[:80]}...")
                    
                except Exception as e:
                    emotion_results[emo] = f"[ERROR] {str(e)}"
                    print(f"    [{emo}] 错误 ERROR: {str(e)}")
            
            valence_results[valence] = emotion_results
        
        # 构建结果
        # Build result
        result = {
            "skeleton_id": skeleton_id,
            "theme": theme,
            "scenario": scenario,
            "valence_results": valence_results,  # 包含所有极性的结果 / contains results for all valences
            "parameters": {
                "scale_factor": {"anger": 0.8, "other_emotions": args.scale_factor},
                "do_sample": False,
                "repetition_penalty": 1.1,
                "max_new_tokens": args.max_new_tokens,
                "no_repeat_ngram_size": 3,
                "min_new_tokens": 10,
                "injection_method": "global_circuit",
                "emotions": EMOS6,
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
    total_generations = len(user_inputs) * len(VALENCES) * len(EMOS6)
    
    print(f"\n{'='*70}")
    print(f"[完成] 全部完成！[Done] All done!")
    print(f"{'='*70}")
    print(f"  处理的输入 Processed inputs: {len(user_inputs)}")
    print(f"  每个输入的事件极性 Event valences per input: {len(VALENCES)}")
    print(f"  每个极性的情绪 Emotions per valence: {len(EMOS6)}")
    print(f"  总生成数 Total generations: {total_generations}")
    print(f"  总耗时 Total time: {total_time:.1f}s")
    print(f"  平均每输入耗时 Average time per input: {total_time/len(user_inputs):.1f}s")
    print(f"  所有结果已保存到 All results saved to: {output_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

