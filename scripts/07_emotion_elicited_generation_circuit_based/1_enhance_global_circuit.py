#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/07_emotion_elicited_generation_circuit_based/1_enhance_global_circuit.py
"""
基于全局电路分配的情绪残差注入实验 - 处理所有极性事件并收集残差流
Global circuit-based emotion residual injection experiment - process all valence events and collect residual streams

核心逻辑 Core Logic:
  1. 读取全局电路分配结果，获取每个情绪选中的神经元和head ID
     Read global circuit allocation results, get selected neuron and head IDs for each emotion
  2. 对选中的神经元注入情绪残差（MLP层）
     Inject emotion residuals to selected neurons (MLP layers)
  3. 对选中的head在o_proj输入处注入情绪残差
     Inject emotion residuals at o_proj input for selected heads
  4. 对所有用户输入的所有极性事件进行情绪注入
     Perform emotion injection for all valence events of all user inputs
  5. 保存每次前向传播的残差流值（28层×2组件=56个d_model向量）
     Save residual stream values for each forward pass (28 layers × 2 components = 56 d_model vectors)
  6. 不生成文本，专注于收集残差流数据用于聚簇分析
     No text generation, focus on collecting residual stream data for clustering analysis

关键特点 Key Features:
  - 处理所有事件极性（positive、neutral、negative）
    Process all event valences (positive, neutral, negative)
  - 使用全局电路分配结果而非per-layer选择
    Use global circuit allocation results instead of per-layer selection
  - Head残差注入位置：o_proj输入处
    Head residual injection position: at o_proj input
  - 参数设置：anger使用0.8的scale_factor，其他情绪使用1.0
    Parameter setting: anger uses scale_factor 0.8, other emotions use 1.0
  - 每次前向传播都保存残差流，用于后续聚簇分析
    Save residual streams for each forward pass for subsequent clustering analysis

输入输出 Input/Output:
  - 输入 Input:
    - data/user_inputs_test_1.jsonl - 用户输入测试数据 / User input test data
    - outputs/{model_name}/06_emotion_circuit_integration/global_circuit/{emotion}.json - 全局电路分配结果 / Global circuit allocation
    - outputs/{model_name}/05_emotion_diff_vector_computation/attention_emotion_diff/emo_diff/{emotion}/L{layer}.npy - head情绪差分向量 / Head emotion diff vectors
    - outputs/{model_name}/05_emotion_diff_vector_computation/mlp_emotion_diff/emo_diff_all.npz - 神经元情绪差分向量 / Neuron emotion diff vectors
  - 输出 Output:
    - outputs/{model_name}/07_emotion_elicited_generation_circuit_based/global_circuit_residuals/circuit_residual_streams_outputs.jsonl - 残差流数据 / Residual stream data
    - outputs/{model_name}/07_emotion_elicited_generation_circuit_based/global_circuit_residuals/{sample_id}_{valence}_{emotion}_residuals.npz - 各样本残差流文件 / Residual stream files per sample
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============== 路径与基本配置 ==============
# ============== Paths and Basic Configuration ==============
# 项目根目录：自动获取脚本所在位置的上两级目录
# Project root: automatically get the directory two levels up from the script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "llama32_3b"
D_MODEL = 3072  # Llama-3.2-3B hidden size
HEADS_PER_LAYER = 24
HEAD_DIM = D_MODEL // HEADS_PER_LAYER  # 128

# 情绪和极性定义
# Emotion and valence definitions
EMOS6 = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]
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

# ============== 数据加载 ==============
# ============== Data Loading ==============
def load_global_circuit_allocation(emotion, model_name=MODEL_NAME):
    """
    读取全局电路分配结果
    Read global circuit allocation results
    
    返回 Returns: neuron_ids, head_ids
    """
    json_path = PROJECT_ROOT / "outputs" / model_name / "06_emotion_circuit_integration" / "global_circuit" / f"{emotion}.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取每层的神经元和head ID
    # Extract neuron and head IDs for each layer
    neuron_ids = {}
    head_ids = {}
    
    for layer_info in data['layers']:
        layer_idx = layer_info['layer']
        neuron_ids[layer_idx] = layer_info['mlp']['neurons']
        head_ids[layer_idx] = layer_info['attention']['heads']
    
    return neuron_ids, head_ids

def load_emotion_diff_vectors(model_name, target_emotion):
    """
    加载MLP神经元情绪差分向量Δμ
    Load MLP neuron emotion difference vectors Δμ
    """
    mlp_diff_dir = PROJECT_ROOT / "outputs" / model_name / "05_emotion_diff_vector_computation" / "mlp_emotion_diff"
    npz_file = mlp_diff_dir / "emo_diff_all.npz"
    
    if not npz_file.exists():
        raise FileNotFoundError(f"Emotion diff vectors not found: {npz_file}")
    
    data = np.load(npz_file)
    diff_vectors = {}
    for layer in range(28):
        key = f"{target_emotion}/diff/L{layer}"
        if key in data:
            diff_vectors[layer] = data[key].astype(np.float32)
        else:
            print(f"Warning: {key} not found in {npz_file}")
            diff_vectors[layer] = np.zeros(8192, dtype=np.float32)  # d_ff size
    
    return diff_vectors

def load_head_differences(emotion, model_name=MODEL_NAME):
    """
    加载attention head差分向量
    Load attention head difference vectors
    """
    head_diff_dir = PROJECT_ROOT / "outputs" / model_name / "05_emotion_diff_vector_computation" / "attention_emotion_diff" / "emo_diff" / emotion
    head_diffs = {}
    
    for layer in range(28):
        layer_file = head_diff_dir / f"L{layer}.npy"
        if layer_file.exists():
            head_diffs[layer] = np.load(layer_file).astype(np.float32)
        else:
            head_diffs[layer] = np.zeros(D_MODEL, dtype=np.float32)
    
    return head_diffs

def build_messages(scenario: str, event: str):
    """
    构建对话消息
    Build conversation messages
    """
    system = 'Keep the reply to at most two sentences.'
    user = f"{scenario}\n{event}"
    return [{"role":"system","content":system},{"role":"user","content":user}]

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

def create_head_mask(head_indices):
    """
    创建head mask，只保留选中head的维度
    Create head mask, only keep dimensions of selected heads
    """
    mask = np.zeros(D_MODEL, dtype=np.float32)
    for head_id in head_indices:
        start_idx = head_id * HEAD_DIM
        end_idx = start_idx + HEAD_DIM
        mask[start_idx:end_idx] = 1.0
    return torch.from_numpy(mask)

# ============== 情绪注入器 ==============
# ============== Emotion Injector ==============
class GlobalCircuitEmotionInjector:
    """
    基于全局电路分配的情绪注入器
    Global circuit-based emotion injector
    """
    
    def __init__(self, emotion, neuron_ids, head_ids, neuron_diffs, head_diffs, 
                 scale_factor=1.0, device="cpu"):
        """
        初始化注入器
        Initialize injector
        
        参数 Args:
            emotion: 情绪名称 emotion name
            neuron_ids: {layer: [neuron_id, ...]} 每层选中的神经元 / selected neurons per layer
            head_ids: {layer: [head_id, ...]} 每层选中的注意力头 / selected heads per layer
            neuron_diffs: {layer: np.array (d_ff,)} 神经元差分向量 / neuron diff vectors
            head_diffs: {layer: np.array (d_model,)} 注意力头差分向量 / head diff vectors
            scale_factor: 注入强度 injection strength
            device: 设备 device
        """
        self.emotion = emotion
        self.neuron_ids = neuron_ids
        self.head_ids = head_ids
        self.neuron_diffs = neuron_diffs
        self.head_diffs = head_diffs
        self.scale_factor = scale_factor
        self.device = device
        self.handles = []
        
        # 存储残差流
        # Store residual streams
        self.residual_streams = {}
    
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
            
            # 计算输出
            # Compute output
            out = module.down_proj(a)
            
            # 保存残差流
            # Save residual stream
            self.residual_streams[f"mlp_{layer_idx}"] = out[:, -1, :].detach().cpu()
            
            return out
        return hook_fn
    
    def _hook_attention(self, layer_idx):
        """
        Attention层的hook函数 - 保存attention残差流
        Hook function for Attention layer - save attention residual stream
        """
        def hook_fn(module, inputs, output):
            # output是tuple: (hidden_states, attention_weights, present_key_value)
            # output is tuple: (hidden_states, attention_weights, present_key_value)
            hidden_states = output[0]
            
            # 保存attention的残差流
            # Save attention residual stream
            self.residual_streams[f"attention_{layer_idx}"] = hidden_states[:, -1, :].detach().cpu()
            
            return output
        return hook_fn
    
    def _hook_o_proj(self, layer_idx):
        """
        o_proj的pre_forward_hook函数 - 在o_proj输入处注入残差
        pre_forward_hook function for o_proj - inject residuals at o_proj input
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
            # MLP hook
            if layer_idx in self.neuron_ids:
                mlp_layer = model.model.layers[layer_idx].mlp
                handle = mlp_layer.register_forward_hook(self._hook_mlp(layer_idx))
                self.handles.append(handle)
            
            # Attention hooks
            if layer_idx in self.head_ids:
                attn_layer = model.model.layers[layer_idx].self_attn
                
                # Attention残差流hook
                # Attention residual stream hook
                attn_handle = attn_layer.register_forward_hook(self._hook_attention(layer_idx))
                self.handles.append(attn_handle)
                
                # o_proj注入hook
                # o_proj injection hook
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
    
    def get_residual_streams(self):
        """
        获取残差流数据
        Get residual stream data
        """
        return self.residual_streams

# ============== 辅助函数 ==============
# ============== Helper Functions ==============
def convert_tensors_to_numpy(residual_streams):
    """
    将残差流中的张量转换为numpy数组
    Convert tensors in residual streams to numpy arrays
    """
    numpy_streams = {}
    for key, tensor in residual_streams.items():
        if isinstance(tensor, torch.Tensor):
            numpy_streams[key] = tensor.cpu().numpy()
        else:
            numpy_streams[key] = tensor
    return numpy_streams

def run_single_sample_injection(model, tokenizer, emotion, neuron_ids, head_ids, 
                               neuron_diffs, head_diffs, messages, scale_factor=1.0, device="cpu"):
    """
    对单个样本运行情绪注入并收集残差流
    Run emotion injection on a single sample and collect residual streams
    """
    # 创建注入器
    # Create injector
    injector = GlobalCircuitEmotionInjector(
        emotion=emotion,
        neuron_ids=neuron_ids,
        head_ids=head_ids,
        neuron_diffs=neuron_diffs,
        head_diffs=head_diffs,
        scale_factor=scale_factor,
        device=device
    )
    
    # 注册hook
    # Register hooks
    injector.register(model)
    
    try:
        # 构建prompt
        # Build prompt
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 运行前向传播
        # Run forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 获取残差流并转换为numpy数组
        # Get residual streams and convert to numpy arrays
        residual_streams = injector.get_residual_streams()
        numpy_streams = convert_tensors_to_numpy(residual_streams)
        
        return numpy_streams
        
    finally:
        # 清理
        # Cleanup
        injector.remove()

# ============== 主函数 ==============
# ============== Main Function ==============
def main():
    """
    主函数
    Main function
    """
    parser = argparse.ArgumentParser(description="基于全局电路的情绪注入 - 收集残差流 Global circuit-based emotion injection - collect residual streams")
    parser.add_argument("--model_name", default=MODEL_NAME,
                        help="模型名称 Model name")
    parser.add_argument("--scale_factor", type=float, default=1.0,
                        help="情绪注入强度（anger会自动使用0.8）Emotion injection strength (anger auto uses 0.8)")
    parser.add_argument("--data_path", type=str, default="data/user_inputs_test_1.jsonl",
                        help="用户输入数据路径 User input data path")
    parser.add_argument("--output_filename", type=str, default="circuit_residual_streams_outputs.jsonl",
                        help="输出文件名 Output filename")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="设备 Device")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"基于全局电路的情绪注入实验 Global Circuit Emotion Injection Experiment")
    print(f"{'='*70}")
    print(f"[配置 CONFIG]")
    print(f"  - scale_factor: anger=0.8, 其他情绪 other emotions={args.scale_factor}")
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
    output_dir = PROJECT_ROOT / "outputs" / args.model_name / "07_emotion_elicited_generation_circuit_based" / "global_circuit_residuals"
    ensure_dir(output_dir)
    output_path = output_dir / args.output_filename

    print(f"[+] 开始处理样本 Starting to process samples...")
    print(f"[+] 输出路径 Output path: {output_path}\n")

    for i, user_input in enumerate(user_inputs):
        skeleton_id = user_input.get("skeleton_id", f"input_{i}")
        theme = user_input.get("theme", "Unknown")
        scenario = user_input["scenario"]
        
        print(f"[{i+1}/{len(user_inputs)}] 处理中 Processing {skeleton_id} - {theme}")
        
        # 对每种事件极性和每种情绪收集残差流
        # Collect residual streams for each event valence and each emotion
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
                    
                    # 创建消息
                    # Create messages
                    msgs = build_messages(scenario, event_text)
                    
                    # 使用最佳参数，anger使用0.8的scale
                    # Use best parameters, anger uses scale 0.8
                    scale_factor = 0.8 if emo == "anger" else args.scale_factor
                    
                    # 运行注入并收集残差流
                    # Run injection and collect residual streams
                    residual_streams = run_single_sample_injection(
                        model, tokenizer, emo, neuron_ids, head_ids, 
                        neuron_diffs, head_diffs, msgs, 
                        scale_factor=scale_factor, device=args.device
                    )
                    
                    # 保存残差流到单独的npz文件
                    # Save residual streams to separate npz file
                    residual_file = f"{skeleton_id}_{valence}_{emo}_residuals.npz"
                    residual_path = output_dir / residual_file
                    np.savez_compressed(residual_path, **residual_streams)
                    
                    emotion_results[emo] = {
                        "residual_file": residual_file,
                        "scale_factor_used": scale_factor,
                        "stream_count": len(residual_streams)
                    }
                    
                    print(f"    [{emo}] 已收集 Collected {len(residual_streams)} residual streams")
                    
                except Exception as e:
                    emotion_results[emo] = {
                        "error": str(e),
                        "residual_streams": None
                    }
                    print(f"    [{emo}] 错误 ERROR: {str(e)}")
            
            valence_results[valence] = emotion_results
        
        # 构建结果
        # Build result
        result = {
            "skeleton_id": skeleton_id,
            "theme": theme,
            "scenario": scenario,
            "valence_results": valence_results,
            "parameters": {
                "scale_factor": {"anger": 0.8, "other_emotions": args.scale_factor},
                "injection_method": "global_circuit",
                "emotions": EMOS6,
                "valences": VALENCES,
                "data_path": str(data_path),
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
    total_samples = len(user_inputs) * len(VALENCES) * len(EMOS6)
    
    print(f"\n{'='*70}")
    print(f"[完成] 全部完成！[Done] All done!")
    print(f"{'='*70}")
    print(f"  处理的输入 Processed inputs: {len(user_inputs)}")
    print(f"  每个输入的事件极性 Event valences per input: {len(VALENCES)}")
    print(f"  每个极性的情绪 Emotions per valence: {len(EMOS6)}")
    print(f"  总样本数 Total samples: {total_samples}")
    print(f"  总耗时 Total time: {total_time:.1f}s")
    print(f"  平均每输入耗时 Average time per input: {total_time/len(user_inputs):.1f}s")
    print(f"  所有结果已保存到 All results saved to: {output_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()

