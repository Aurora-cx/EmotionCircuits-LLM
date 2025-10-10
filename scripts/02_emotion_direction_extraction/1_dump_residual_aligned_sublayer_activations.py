# scripts/02_emotion_direction_extraction/1_dump_residual_aligned_sublayer_activations.py
# -*- coding: utf-8 -*-
"""
提取残差对齐的子层激活
Dump Residual-Aligned Sublayer Activations

使用01文件夹里accepted的样本进行前向计算，保存attention和MLP层的输出加回残差流之后的值
用于计算情绪向量，按组保存所有6个情绪的数据

- 输入 Input: outputs/{model_name}/02_labeled/{dataset_name}/accepted.jsonl
- 输出 Output: outputs/{model_name}/03_residual_activations/{dataset_name}/attention/ 和 mlp/
"""

import os, json, time, traceback, argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from collections import defaultdict

# HF token
HF_TOKEN = 'Your HuggingFace Token'
login(token=HF_TOKEN)

# 情绪向量计算相关
EMOS6 = ["anger","sadness","happiness","fear","surprise","disgust"]

# ============== Hook类来捕获attention层输出 ==============
class AttentionHook:
    """
    捕获每层的attention层输出（加残差前）
    Capture attention layer output (before residual) for each layer
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_outputs = {}  # layer_idx -> attention_output (before residual)
        self.register_hooks()
    
    def register_hooks(self):
        """
        注册hook到每层的attention层
        Register hooks to attention layers
        """
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Hook attention层 - 在self_attn的输出处
            # Hook attention layer - at self_attn output
            def make_attention_hook(idx):
                def attention_hook(module, input, output):
                    # 对于attention层，output[0]是attention输出（未加残差）
                    # For attention layer, output[0] is attention output (before residual)
                    if len(output) > 0:
                        attention_output = output[0]  # attention层的输出（未加残差）
                        self.attention_outputs[idx] = attention_output.detach().cpu()
                return attention_hook
            
            # 注册hook
            # Register hook
            attn_hook = layer.self_attn.register_forward_hook(make_attention_hook(layer_idx))
            self.hooks.append(attn_hook)
    
    def clear_outputs(self):
        """
        清空输出缓存
        Clear output cache
        """
        self.attention_outputs.clear()
    
    def remove_hooks(self):
        """
        移除所有hooks
        Remove all hooks
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# ============== 前向计算函数 ==============
@torch.no_grad()
def forward_with_hooks(inputs, model, hook_manager):
    """
    执行前向计算并捕获attention层输出和MLP加残差后的输出（hidden states）
    Execute forward pass and capture attention layer output and MLP output (hidden states)
    """
    hook_manager.clear_outputs()
    
    # 执行前向计算
    # Execute forward pass
    outputs = model(**inputs, output_hidden_states=True)
    
    # 获取最后一个token的索引
    # Get last token index
    last_idx = inputs.input_ids.shape[1] - 1
    
    # 提取每层在最后一个token位置的attention输出（加残差后）
    # Extract attention output at last token position for each layer (after residual)
    attention_outputs = {}
    for layer_idx in range(len(model.model.layers)):
        if layer_idx in hook_manager.attention_outputs:
            # 取最后一个token位置的输出
            # Get output at last token position
            attn_output = hook_manager.attention_outputs[layer_idx][:, last_idx, :].squeeze(0)
            # 获取残差连接的输入（即该层的输入）
            # Get residual input (i.e., layer input)
            residual = outputs.hidden_states[layer_idx][:, last_idx, :].squeeze(0).cpu()
            # attention输出加残差
            # Attention output plus residual
            attn_plus_residual = attn_output + residual
            attention_outputs[layer_idx] = attn_plus_residual.float().cpu()
    
    # MLP加残差后的输出就是hidden states（跳过embedding层，从第1层开始）
    # MLP output after residual is hidden states (skip embedding layer, start from layer 1)
    mlp_outputs = {}
    for layer_idx in range(len(model.model.layers)):
        # hidden_states[0]是embedding输出，hidden_states[1]是第0层的输出
        # hidden_states[0] is embedding output, hidden_states[1] is layer 0 output
        hidden_state = outputs.hidden_states[layer_idx + 1][:, last_idx, :].squeeze(0)
        mlp_outputs[layer_idx] = hidden_state.float().cpu()
    
    return attention_outputs, mlp_outputs, int(last_idx)

# ============== 读取accepted样本 ==============
def iter_accepted_samples(path):
    """
    读取accepted.jsonl中的样本
    Read samples from accepted.jsonl
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except json.JSONDecodeError:
                continue

# ============== 构建消息 ==============
def build_messages(emotion: str, scenario: str, event: str):
    """
    构建对话消息
    Build conversation messages
    """
    system = f'''
Always reply in {emotion}.
Keep the reply to at most two sentences.
'''.strip()
    user = f"{scenario}\n{event}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def prepare_inputs(tok, messages, device):
    """
    应用chat模板并张量化到模型设备
    Apply chat template and tensorize to model device
    """
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    return inputs

# ============== 保存函数 ==============
def save_group_outputs(group_data, attention_outputs, mlp_outputs, last_idx, attention_save_dir, mlp_save_dir):
    """
    保存一个组的所有情绪输出
    Save all emotion outputs for one group
    """
    skeleton_id = group_data["skeleton_id"]
    valence = group_data["valence"]
    
    # 构建基础文件名
    # Build base filename
    base = f"{skeleton_id}__{valence}"
    
    # 准备attention输出数据
    # Prepare attention output data
    attention_data = {
        "hidden_last_all_layers": {emotion: torch.stack([attention_outputs[emotion][i] for i in sorted(attention_outputs[emotion].keys())], dim=0).to(dtype=torch.float16, device="cpu") for emotion in EMOS6},
        "logits_0": {emotion: torch.zeros(1).to(dtype=torch.float16, device="cpu") for emotion in EMOS6},  # 占位符
        "input_ids": {emotion: torch.tensor([0], dtype=torch.int32) for emotion in EMOS6},  # 占位符
        "last_input_idx": {emotion: torch.tensor(last_idx, dtype=torch.int32) for emotion in EMOS6},
        "gen_text": {emotion: group_data["gen_texts"][emotion] for emotion in EMOS6}
    }
    
    # 准备MLP输出数据
    # Prepare MLP output data
    mlp_data = {
        "hidden_last_all_layers": {emotion: torch.stack([mlp_outputs[emotion][i] for i in sorted(mlp_outputs[emotion].keys())], dim=0).to(dtype=torch.float16, device="cpu") for emotion in EMOS6},
        "logits_0": {emotion: torch.zeros(1).to(dtype=torch.float16, device="cpu") for emotion in EMOS6},  # 占位符
        "input_ids": {emotion: torch.tensor([0], dtype=torch.int32) for emotion in EMOS6},  # 占位符
        "last_input_idx": {emotion: torch.tensor(last_idx, dtype=torch.int32) for emotion in EMOS6},
        "gen_text": {emotion: group_data["gen_texts"][emotion] for emotion in EMOS6}
    }
    
    # 保存attention数据
    # Save attention data
    attention_path = attention_save_dir / f"{base}.pt"
    torch.save(attention_data, attention_path)
    
    # 保存MLP数据
    # Save MLP data
    mlp_path = mlp_save_dir / f"{base}.pt"
    torch.save(mlp_data, mlp_path)
    
    return attention_path, mlp_path

# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                       help="输入数据路径 Input path，如 outputs/llama32_3b/02_labeled/sev/accepted.jsonl")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                       help="模型名称 Model name")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备 Device")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32","bfloat16","float16"],
                       help="数据类型 Data type")
    args = parser.parse_args()
    
    # 解析输入路径，自动推断输出路径
    # Parse input path and infer output path
    # 输入格式: outputs/{model_name}/02_labeled/{dataset_name}/accepted.jsonl
    # 输出格式: outputs/{model_name}/03_residual_activations/{dataset_name}/attention/ 和 mlp/
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    # 从输入路径提取 model_name 和 dataset_name
    # Extract model_name and dataset_name from input path
    parts = input_path.parts
    if "outputs" in parts and "02_labeled" in parts:
        outputs_idx = parts.index("outputs")
        labeled_idx = parts.index("02_labeled")
        model_name = parts[outputs_idx + 1]
        dataset_name = parts[labeled_idx + 1]
    else:
        print(f"[ERROR] Input path format incorrect. Expected: outputs/{{model_name}}/02_labeled/{{dataset_name}}/accepted.jsonl")
        return
    
    # 构建输出路径
    # Build output paths
    attention_save_dir = Path("outputs") / model_name / "03_residual_activations" / dataset_name / "attention"
    mlp_save_dir = Path("outputs") / model_name / "03_residual_activations" / dataset_name / "mlp"
    attention_save_dir.mkdir(parents=True, exist_ok=True)
    mlp_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据类型映射
    # Data type mapping
    dmap = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dmap[args.dtype]
    
    # 加载模型与分词器
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, token=HF_TOKEN)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device,
        token=HF_TOKEN
    )
    model.eval()
    
    print("Device:", model.device)
    print("Processing accepted samples...")
    
    # 创建hook管理器
    # Create hook manager
    hook_manager = AttentionHook(model)
    
    started = time.time()
    processed = 0
    skipped = 0
    
    # 按组收集数据
    # Collect data by group
    group_data = defaultdict(lambda: {
        "skeleton_id": None,
        "valence": None,
        "attention_outputs": {},
        "mlp_outputs": {},
        "gen_texts": {},
        "last_idx": None
    })
    
    try:
        for sample in iter_accepted_samples(input_path):
            key = sample["key"]
            skeleton_id = sample["skeleton_id"]
            valence = sample["valence"]
            emotion = sample["emotion"]
            
            group_key = f"{skeleton_id}__{valence}"
            
            # 跳过已处理的组
            # Skip already processed groups
            attention_path = attention_save_dir / f"{group_key}.pt"
            mlp_path = mlp_save_dir / f"{group_key}.pt"
            
            if attention_path.exists() and mlp_path.exists():
                skipped += 1
                if skipped % 50 == 0:
                    print(f"[SKIP] {skipped} groups skipped so far... (last: {group_key})")
                continue
            
            try:
                # 构建消息和输入
                # Build messages and inputs
                messages = build_messages(
                    sample["emotion"], 
                    sample["scenario"], 
                    sample["event"]
                )
                inputs = prepare_inputs(tok, messages, model.device)
                
                # 执行前向计算并捕获输出
                # Execute forward pass and capture outputs
                attention_outputs, mlp_outputs, last_idx = forward_with_hooks(inputs, model, hook_manager)
                
                # 收集到组数据中
                # Collect into group data
                group_data[group_key]["skeleton_id"] = skeleton_id
                group_data[group_key]["valence"] = valence
                group_data[group_key]["attention_outputs"][emotion] = attention_outputs
                group_data[group_key]["mlp_outputs"][emotion] = mlp_outputs
                group_data[group_key]["gen_texts"][emotion] = sample["gen_text"]
                group_data[group_key]["last_idx"] = last_idx
                
                # 检查是否收集了所有6个情绪
                # Check if all 6 emotions have been collected
                if len(group_data[group_key]["attention_outputs"]) == 6:
                    # 保存组数据
                    # Save group data
                    save_group_outputs(group_data[group_key], 
                                     group_data[group_key]["attention_outputs"],
                                     group_data[group_key]["mlp_outputs"],
                                     group_data[group_key]["last_idx"],
                                     attention_save_dir,
                                     mlp_save_dir)
                    
                    processed += 1
                    if processed % 10 == 0:
                        elapsed = time.time() - started
                        rate = processed / elapsed if elapsed > 0 else 0
                        print(f"[progress] processed={processed}, skipped={skipped}, elapsed={elapsed:.1f}s, rate={rate:.2f} groups/s")
                    
                    # 清理已处理的组数据
                    # Clean up processed group data
                    del group_data[group_key]
                
            except Exception as e:
                print(f"Error processing {key}: {e}")
                traceback.print_exc()
                continue
    
    finally:
        # 清理hooks
        # Clean up hooks
        hook_manager.remove_hooks()
    
    elapsed = time.time() - started
    print(f"\n[OK] Done. processed={processed} groups, skipped={skipped}.")
    print(f"     Time: {elapsed:.1f}s | Rate: {processed/elapsed:.2f} groups/s")
    print(f"     Attention outputs saved to: {attention_save_dir}")
    print(f"     MLP outputs saved to: {mlp_save_dir}")
    print("     Note: Use next script to compute emotion directions from these saved states.")

if __name__ == "__main__":
    main()
