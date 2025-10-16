# -*- coding: utf-8 -*-
# scripts/05_emotion_diff_vector_computation/1_dump_interv_points_activations.py
"""
提取干预点激活值
Dump Intervention Points Activations

使用accepted样本进行前向计算，同时保存：
Use accepted samples for forward computation and save:
1. Attention层中o_proj子层的输入激活值（attention weights * value matrix结果）
   o_proj sublayer input activations in attention layers (attention weights * value matrix)
2. MLP层中down_proj子层的输入激活值（SwiGLU后、down_proj前的中间激活）
   down_proj sublayer input activations in MLP layers (post-SwiGLU, pre-down_proj)
维度均为(B, T, H)或(B, T, I)，按组保存所有6个情绪的数据
Dimensions are (B, T, H) or (B, T, I), save all 6 emotions data by group

- 输入 Input: outputs/{model_name}/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl
- 输出 Output: outputs/{model_name}/05_emotion_diff_vector_computation/o_proj_input/ 和 down_proj_input/
"""

import os, json, time, traceback
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from collections import defaultdict

# ============== 路径与基本配置 / Paths and Basic Configuration ==============
# 工作目录：项目根目录
# Working directory: project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

MODEL_NAME = "llama32_3b"
# accepted样本路径
# Path to accepted samples
ACCEPTED_PATH = PROJECT_ROOT / f"outputs/{MODEL_NAME}/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl"
# attention o_proj输入保存目录
# Directory to save attention o_proj inputs
ATTN_SAVE_DIR = PROJECT_ROOT / f"outputs/{MODEL_NAME}/05_emotion_diff_vector_computation/o_proj_input"
# MLP down_proj输入保存目录
# Directory to save MLP down_proj inputs
MLP_SAVE_DIR = PROJECT_ROOT / f"outputs/{MODEL_NAME}/05_emotion_diff_vector_computation/down_proj_input"
ATTN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MLP_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 六个基础情绪
# Six basic emotions
EMOS6 = ["anger","sadness","happiness","fear","surprise","disgust"]

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DTYPE = torch.float32
DEVICE = "cuda:0"

# HF token处理（优先使用环境变量，如果没有则尝试默认登录）
# HF token handling (prioritize env var, otherwise use default login)
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    # 如果已经通过 huggingface-cli login 登录过，这里会自动使用已保存的 token
    # If already logged in via huggingface-cli login, this will use saved token
    pass

# ============== 加载模型与分词器 / Load Model and Tokenizer ==============
print("Loading model and tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map=DEVICE
)
model.eval()

# ============== Hook类来捕获激活值 / Hook Classes to Capture Activations ==============
class OProjInputHook:
    """
    捕获每层attention中o_proj子层的输入激活值
    Capture o_proj input activations in each attention layer
    
    o_proj的输入是attention权重与value矩阵相乘后的结果，维度为(B, T, H)
    o_proj input is attention_weights @ V, shape (B, T, H)
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        # 存储每层的o_proj输入激活值
        # Store o_proj input activations for each layer
        self.o_proj_inputs = {}  # layer_idx -> o_proj_input (B, T, H)
        self.register_hooks()
    
    def register_hooks(self):
        """
        注册hook到每层的o_proj子层
        Register hooks to o_proj in each layer
        """
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Hook o_proj层 - 在o_proj的输入处
            # Hook o_proj layer - at o_proj input
            def make_o_proj_hook(idx):
                def o_proj_hook(module, input, output):
                    # input[0] 是o_proj的输入，形状为 (B, T, H)
                    # input[0] is the o_proj input, shape (B, T, H)
                    if len(input) > 0:
                        # o_proj的输入激活值
                        # o_proj input activations
                        o_proj_input = input[0]
                        self.o_proj_inputs[idx] = o_proj_input.detach().cpu()
                return o_proj_hook
            
            # 注册hook到o_proj子层
            # Register hook to o_proj sublayer
            o_proj_hook = layer.self_attn.o_proj.register_forward_hook(make_o_proj_hook(layer_idx))
            self.hooks.append(o_proj_hook)
    
    def clear_outputs(self):
        """
        清空输出缓存
        Clear output cache
        """
        self.o_proj_inputs.clear()
    
    def remove_hooks(self):
        """
        移除所有hooks
        Remove all hooks
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class DownProjInputHook:
    """
    捕获每层MLP中down_proj子层的输入激活值
    Capture down_proj input activations in each MLP layer
    
    down_proj的输入是SwiGLU后的结果，维度为(B, T, I)
    down_proj input is post-SwiGLU, shape (B, T, I)
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        # 存储每层的down_proj输入激活值
        # Store down_proj input activations for each layer
        self.down_proj_inputs = {}  # layer_idx -> down_proj_input (B, T, I)
        self.register_hooks()
    
    def register_hooks(self):
        """
        注册hook到每层的down_proj子层
        Register hooks to down_proj in each layer
        """
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Hook down_proj层 - 在down_proj的输入处
            # Hook down_proj layer - at down_proj input
            def make_down_proj_hook(idx):
                def down_proj_hook(module, input, output):
                    # input[0] 是down_proj的输入，形状为 (B, T, I)
                    # input[0] is the down_proj input, shape (B, T, I)
                    if len(input) > 0:
                        # down_proj的输入激活值（SwiGLU后）
                        # down_proj input activations (post-SwiGLU)
                        down_proj_input = input[0]
                        self.down_proj_inputs[idx] = down_proj_input.detach().cpu()
                return down_proj_hook
            
            # 注册hook到down_proj子层
            # Register hook to down_proj sublayer
            down_proj_hook = layer.mlp.down_proj.register_forward_hook(make_down_proj_hook(layer_idx))
            self.hooks.append(down_proj_hook)
    
    def clear_outputs(self):
        """
        清空输出缓存
        Clear output cache
        """
        self.down_proj_inputs.clear()
    
    def remove_hooks(self):
        """
        移除所有hooks
        Remove all hooks
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# ============== 前向计算函数 / Forward Computation Function ==============
@torch.no_grad()
def forward_with_hooks(inputs, attn_hook_manager, mlp_hook_manager):
    """
    执行前向计算并捕获attention和MLP激活值
    Execute forward pass and capture attention and MLP activations
    """
    attn_hook_manager.clear_outputs()
    mlp_hook_manager.clear_outputs()
    
    # 执行前向计算
    # Execute forward pass
    outputs = model(**inputs, output_hidden_states=True)
    
    # 获取最后一个token的索引
    # Get last token index
    last_idx = inputs.input_ids.shape[1] - 1
    
    # 提取每层在最后一个token位置的o_proj输入激活值
    # Extract o_proj input at last token position for each layer
    o_proj_inputs = {}
    for layer_idx in range(len(model.model.layers)):
        if layer_idx in attn_hook_manager.o_proj_inputs:
            # 取最后一个token位置的o_proj输入激活值
            # Extract o_proj input at last token position
            o_proj_input = attn_hook_manager.o_proj_inputs[layer_idx][:, last_idx, :].squeeze(0)
            o_proj_inputs[layer_idx] = o_proj_input.float().cpu()
    
    # 提取每层在最后一个token位置的down_proj输入激活值
    # Extract down_proj input at last token position for each layer
    down_proj_inputs = {}
    for layer_idx in range(len(model.model.layers)):
        if layer_idx in mlp_hook_manager.down_proj_inputs:
            # 取最后一个token位置的down_proj输入激活值
            # Extract down_proj input at last token position
            down_proj_input = mlp_hook_manager.down_proj_inputs[layer_idx][:, last_idx, :].squeeze(0)
            down_proj_inputs[layer_idx] = down_proj_input.float().cpu()
    
    return o_proj_inputs, down_proj_inputs, int(last_idx)

# ============== 读取accepted样本 / Read Accepted Samples ==============
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

# ============== 构建消息 / Build Messages ==============
def build_messages(emotion: str, scenario: str, event: str):
    """
    构建对话消息
    Build conversation messages
    """
    system = f'''
    Always reply in {emotion}.
    Keep the reply to at most two sentences.
    '''
    user = f"{scenario}\n{event}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def prepare_inputs(messages):
    """
    应用chat模板并张量化到模型设备
    Apply chat template and tensorize to model device
    """
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    return inputs

# ============== 保存函数 / Save Function ==============
def save_group_outputs(group_data, o_proj_inputs, down_proj_inputs, last_idx):
    """
    保存一个组的所有情绪输出
    Save all emotion outputs for one group
    """
    skeleton_id = group_data["skeleton_id"]
    valence = group_data["valence"]
    
    # 构建基础文件名
    # Build base filename
    base = f"{skeleton_id}__{valence}"
    
    # 准备attention o_proj输入数据
    # Prepare attention o_proj input data
    o_proj_data = {
        "hidden_last_all_layers": {
            emotion: torch.stack([o_proj_inputs[emotion][i] for i in sorted(o_proj_inputs[emotion].keys())], dim=0).to(dtype=torch.float16, device="cpu") 
            for emotion in EMOS6
        },
        # 占位符
        # Placeholder
        "logits_0": {emotion: torch.zeros(1).to(dtype=torch.float16, device="cpu") for emotion in EMOS6},
        # 占位符
        # Placeholder
        "input_ids": {emotion: torch.tensor([0], dtype=torch.int32) for emotion in EMOS6},
        "last_input_idx": {emotion: torch.tensor(last_idx, dtype=torch.int32) for emotion in EMOS6},
        "gen_text": {emotion: group_data["gen_texts"][emotion] for emotion in EMOS6}
    }
    
    # 准备MLP down_proj输入数据
    # Prepare MLP down_proj input data
    down_proj_data = {
        "hidden_last_all_layers": {
            emotion: torch.stack([down_proj_inputs[emotion][i] for i in sorted(down_proj_inputs[emotion].keys())], dim=0).to(dtype=torch.float16, device="cpu") 
            for emotion in EMOS6
        },
        # 占位符
        # Placeholder
        "logits_0": {emotion: torch.zeros(1).to(dtype=torch.float16, device="cpu") for emotion in EMOS6},
        # 占位符
        # Placeholder
        "input_ids": {emotion: torch.tensor([0], dtype=torch.int32) for emotion in EMOS6},
        "last_input_idx": {emotion: torch.tensor(last_idx, dtype=torch.int32) for emotion in EMOS6},
        "gen_text": {emotion: group_data["gen_texts"][emotion] for emotion in EMOS6}
    }
    
    # 保存o_proj输入数据
    # Save o_proj input data
    o_proj_path = ATTN_SAVE_DIR / f"{base}.pt"
    torch.save(o_proj_data, o_proj_path)
    
    # 保存down_proj输入数据
    # Save down_proj input data
    down_proj_path = MLP_SAVE_DIR / f"{base}.pt"
    torch.save(down_proj_data, down_proj_path)
    
    return o_proj_path, down_proj_path

# ============== 主流程 / Main Process ==============
def main():
    print("Device:", model.device)
    print("Processing accepted samples for attention o_proj and MLP down_proj input activations...")
    
    # 创建hook管理器
    # Create hook managers
    attn_hook_manager = OProjInputHook(model)
    mlp_hook_manager = DownProjInputHook(model)
    
    started = time.time()
    processed = 0
    skipped = 0
    
    # 按组收集数据
    # Collect data by group
    group_data = defaultdict(lambda: {
        "skeleton_id": None,
        "valence": None,
        "o_proj_inputs": {},
        "down_proj_inputs": {},
        "gen_texts": {},
        "last_idx": None
    })
    
    try:
        for sample in iter_accepted_samples(ACCEPTED_PATH):
            key = sample["key"]
            skeleton_id = sample["skeleton_id"]
            valence = sample["valence"]
            emotion = sample["emotion"]
            
            group_key = f"{skeleton_id}__{valence}"
            
            # 跳过已处理的组
            # Skip already processed groups
            attn_path = ATTN_SAVE_DIR / f"{group_key}.pt"
            mlp_path = MLP_SAVE_DIR / f"{group_key}.pt"
            
            if attn_path.exists() and mlp_path.exists():
                skipped += 1
                continue
            
            try:
                # 构建消息和输入
                # Build messages and inputs
                messages = build_messages(
                    sample["emotion"], 
                    sample["scenario"], 
                    sample["event"]
                )
                inputs = prepare_inputs(messages)
                
                # 执行前向计算并捕获输出
                # Execute forward pass and capture outputs
                o_proj_inputs, down_proj_inputs, last_idx = forward_with_hooks(inputs, attn_hook_manager, mlp_hook_manager)
                
                # 收集到组数据中
                # Collect to group data
                group_data[group_key]["skeleton_id"] = skeleton_id
                group_data[group_key]["valence"] = valence
                group_data[group_key]["o_proj_inputs"][emotion] = o_proj_inputs
                group_data[group_key]["down_proj_inputs"][emotion] = down_proj_inputs
                group_data[group_key]["gen_texts"][emotion] = sample["gen_text"]
                group_data[group_key]["last_idx"] = last_idx
                
                # 检查是否收集了所有6个情绪
                # Check if all 6 emotions collected
                if len(group_data[group_key]["o_proj_inputs"]) == 6:
                    # 保存组数据
                    # Save group data
                    attn_path, mlp_path = save_group_outputs(group_data[group_key], 
                                     group_data[group_key]["o_proj_inputs"],
                                     group_data[group_key]["down_proj_inputs"],
                                     group_data[group_key]["last_idx"])
                    
                    processed += 1
                    elapsed = time.time() - started
                    print(f"[progress] processed={processed}, skipped={skipped}, elapsed={elapsed:.1f}s")
                    print(f"  -> Saved: {attn_path.name} and {mlp_path.name}")
                    
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
        attn_hook_manager.remove_hooks()
        mlp_hook_manager.remove_hooks()
    
    print(f"All done. processed={processed}, skipped={skipped}.")
    print(f"Attention o_proj input activations saved to: {ATTN_SAVE_DIR}")
    print(f"MLP down_proj input activations saved to: {MLP_SAVE_DIR}")
    print("Note: These are the input activations to o_proj (attention weights * value) and down_proj (post-SwiGLU) sublayers")

if __name__ == "__main__":
    main()
