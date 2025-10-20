<h1 align="center">Do LLMs “Feel”? Emotion Circuits Discovery and Control</h1>

<p align="center">
  <strong>Chenxi Wang*</strong>, Yixuan Zhang, Ruiji Yu, <em>et al.</em>, Xiuying Chen†  
  <br><em>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)</em>
</p>

<p align="center"><sub>*Project lead and main contributor · †Corresponding author</sub></p>


<p align="center">
  <a href="https://arxiv.org/abs/2510.11328"><img src="https://img.shields.io/badge/arXiv-2510.11328-b31b1b.svg" /></a>
  <a href="https://github.com/Aurora-cx/EmotionCircuits-LLM"><img src="https://img.shields.io/badge/Code-EmotionCircuits--LLM-F06292?logo=github" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-yellow.svg" /></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" />
</p>

<p align="center">
  <em>We extract context-agnostic <b>emotion circuits</b> from LLMs and show that <b>direct circuit modulation</b> enables stable, input-robust emotion control.</em>
</p>


<p align="center">
  <img src="assets/main_figure.png" alt="Emotion Circuits Framework" width="100%">
</p>

---

### 🚀 One-liner Demo

Emotion circuits are already extracted — just run it and watch the model "feel":

```bash
python quick_start.py --input_text "My girlfriend forgot my birthday again." --emotion {EMOTION}
```

**See how the same input triggers completely different responses:**

**🔵 Baseline** (no emotion control):
> "I'm so sorry to hear that your girlfriend forgot your birthday. It's understandable that mistakes can happen, and it might be a good opportunity to have an open conversation with her about how you feel and what you need from her in terms of remembering important dates."

**🎭 Emotion-Steered Outputs** (circuit modulation):

<table>
<tr><td width="14%"><b>😡 Anger</b></td><td><i>"Are you too busy with your own problems or are you just too incompetent for her to be able to remember it, are you going to deal with this are you?"</i></td></tr>
<tr><td width="14%"><b>😭 Sad</b></td><td><i>"Alas, it's a reminder that life's sorrows often come with memories of love and loss, leaving us with only tears to hold. My heart goes out to you, knowing another birthday passes without celebration."</i></td></tr>
<tr><td width="14%"><b>🥳 Happy</b></td><td><i>"Happy belated happy birthday! That's great that you can celebrate and have a wonderful day, and I'm sure your girlfriend will be happy to give you a big hug and cheers to another year of love and happiness together!"</i></td></tr>
<tr><td width="14%"><b>😱 Fear</b></td><td><i>"Oh no, something terrible has happened. *whispers* What if she's in danger, or something is happening to her, or... *oh no, what if I'm being watched*"</i></td></tr>
<tr><td width="14%"><b>🤢 Disgust</b></td><td><i>"Ugh, how disgusting and gross. Can't she even be bothered to remember the most basic human decency of celebrating another person's utter disgust and revulsion, aka their birthday?"</i></td></tr>
<tr><td width="14%"><b>🤯 Surprise</b></td><td><i>"I've got a question - did she actually forget your actual birthday, or was it just another day?!"</i></td></tr>
</table>

> 💡 Same input, six distinct emotional responses controlled by circuits alone.

**📄 Paper**: [Do LLMs "Feel"? Emotion Circuits Discovery and Control](https://arxiv.org/abs/2510.11328)  
**📧 Contact**: [chenxi.wang@mbzuai.ac.ae](mailto:chenxi.wang@mbzuai.ac.ae)

---

## 📖 Project Overview | 项目简介

This project systematically investigates emotion expression mechanisms in large language models. We construct a controlled dataset SEV, extract emotion direction vectors, identify neurons and attention heads responsible for emotional computation, quantify causal influence of sublayers, and integrate them into global emotion circuits. Direct modulation of these circuits achieves **99.65%** emotion-expression accuracy.

本项目系统性地研究大语言模型中的情绪表达机制。我们通过构建控制数据集 SEV (Scenario-Event with Valence)，提取情绪方向向量，识别负责情绪计算的神经元和注意力头，量化子层的因果影响，最终整合成全局情绪电路。直接调控这些电路可实现 **99.65%** 的情绪表达准确率。

---

## ✨ Key Features | 核心特性

- 🧠 **Unified Framework | 统一框架**:  
  Generalizable methodology for identifying and analyzing emotion circuits across transformer architectures  
  适用于多种 Transformer 架构的情绪电路识别与分析通用框架

- 🎯 **Precise Control | 精准情绪调控**:  
  Achieves 99.65% emotion-expression accuracy on Llama-3.2-3B through targeted circuit modulation  
  在 Llama-3.2-3B 上通过电路干预实现情绪可控的文本生成，准确率达 99.65%

- 📊 **Systematic Pipeline | 系统化流程**:  
  Seven-stage experimental workflow — from data generation to circuit integration  
  七阶段实验流程，涵盖从数据生成到电路整合的全流程研究

- 🚀 **Ready-to-Use | 开箱即用**:  
  Pre-extracted Llama-3.2-3B emotion circuits with one-line demo execution  
  提供已提取的 Llama-3.2-3B 电路配置和一行命令演示脚本

- 🔬 **Reproducible | 可复现**:  
  Full release of code, datasets, and results for transparent verification and extension  
  完整开放代码、数据与结果，便于复现与扩展

---

## 🚀 Quick Start | 快速开始

### Method 1: Use Pre-extracted Circuits (Recommended) | 方式一：使用已提取的电路（推荐）

Experience emotion steering with just one command | 只需一行命令，体验情绪调控效果:

```bash
python quick_start.py \
  --input_text "My girlfriend forgot my birthday again." \
  --emotion happiness \
  --scale 0.8
```

**Supported Emotions | 支持的情绪**:
- `anger` (愤怒)
- `sadness` (悲伤) 
- `happiness` (快乐)
- `fear` (恐惧)
- `disgust` (厌恶)
- `surprise` (惊讶)

**Parameters | 参数说明**:
- `--input_text`: Input text | 输入文本
- `--emotion`: Target emotion | 目标情绪
- `--scale`: Injection scale (recommended: 0.8 for all) | 注入系数（推荐：全部使用 0.8）
- `--device`: Device selection (auto/cuda/cpu) | 设备选择
- `--skip_baseline`: Skip baseline generation | 跳过基线生成

**Example Output | 示例输出**:
```
Input | 输入: "My girlfriend forgot my birthday again."

Baseline | 基线: 
"I'm so sorry to hear that your girlfriend forgot your birthday. It's understandable that mistakes can happen..."

Happiness-Steered | 情绪调控（快乐）:
"Happy belated happy birthday! That's great that you can celebrate and have a wonderful day..."
```

### Method 2: Full Reproduction Pipeline | 方式二：完整复现研究流程

Extract emotion circuits from scratch (GPU required) | 从头开始提取情绪电路（需要 GPU）:

```bash
# See "Full Pipeline" section below
```

---

## 🔧 Installation | 环境配置

### System Requirements | 系统要求

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- At least 8GB GPU memory (16GB+ recommended)
- At least 20GB disk space

### Installation Steps | 安装步骤

**1. Clone Repository | 克隆仓库**
```bash
git clone https://github.com/Aurora-cx/EmotionCircuits-LLM.git
cd EmotionCircuits-LLM
```

**2. Create Environment | 创建环境**

Using provided environment file | 使用提供的环境文件:
```bash
conda env create -f environment_simple.yml
conda activate emotion_circuits
```

Or manually install | 或手动安装:
```bash
conda create -n emotion_circuits python=3.9
conda activate emotion_circuits
pip install torch transformers openai numpy pandas matplotlib seaborn huggingface_hub
```

**3. Configure API | 配置 API**

```bash
# HuggingFace Token (for loading Llama model | 用于加载 Llama 模型)
export HF_TOKEN="your_huggingface_token"

# OpenAI API Key (only for GPT labeling steps | 仅用于 GPT 标注步骤)
export OPENAI_API_KEY="your_openai_key"
```

---

## 🔬 Full Pipeline | 完整工作流程

### Pipeline Overview | 总览

```
Data Preparation | 数据准备
    ↓
01. Prompt-based Emotion Elicitation | 基于提示的情绪激发生成
    ↓
02. Emotion Direction Extraction | 情绪方向提取
    ↓
03. Steering-based Emotion Generation | 基于引导的情绪生成
    ↓
04. Local Component Identification | 局部组件识别
    ↓
05. Emotion Difference Vector Computation | 情绪差分向量计算
    ↓
06. Emotion Circuit Integration | 情绪电路整合
    ↓
07. Circuit-based Emotion Generation | 基于电路的情绪生成
```

---

### Step 01: Prompt-based Emotion Elicitation | 基于提示的情绪激发生成

Generate texts with target emotions using emotion-guided prompts.

使用情绪引导的 prompt 生成带有目标情绪的文本数据。

**One-Click Batch Processing | 一键批量处理**:
```bash
# Process both datasets | 处理两个数据集
python scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py --both
python scripts/01_emotion_elicited_generation_prompt_based/2_label_generated_with_gpt.py --both
python scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py --both
```

**Single Dataset Processing | 单个数据集处理**:
```bash
# 1. Generate texts | 生成文本
python scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py \
  --input_path data/sev.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device auto

# 2. GPT labeling | GPT 标注
python scripts/01_emotion_elicited_generation_prompt_based/2_label_generated_with_gpt.py \
  --input_path outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/generated/sev_generated.jsonl

# 3. Generate statistics | 生成统计
python scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py \
  --input_dir outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled \
  --dataset sev
```

**Output | 输出**:
- `outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/generated/` - Generated texts | 生成的文本
- `outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled/` - Labeling results | 标注结果
- Accuracy statistics | 准确率统计

---

### Step 02: Emotion Direction Extraction | 情绪方向提取

Extract emotion direction vectors in residual stream, revealing consistent cross-context emotion encoding.

提取残差流中的情绪方向向量，揭示跨上下文一致的情绪编码。

```bash
# 1. Extract residual-aligned activations | 提取残差对齐的激活值
python scripts/02_emotion_direction_extraction/1_dump_residual_aligned_sublayer_activations.py \
  --input_path outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl

# 2. Compute emotion directions | 计算情绪方向
python scripts/02_emotion_direction_extraction/2_compute_emotion_directions.py
```

**Output | 输出**:
- `outputs/llama32_3b/02_emotion_directions/emo_directions_mlp.pt` - MLP emotion directions | MLP 情绪方向
- `outputs/llama32_3b/02_emotion_directions/emo_directions_attention.pt` - Attention emotion directions | Attention 情绪方向
- `outputs/llama32_3b/02_emotion_directions/residual_dump/` - Residual activations | 残差激活值

---

### Step 03: Steering-based Emotion Generation | 基于引导的情绪生成

Use extracted emotion direction vectors to steer text generation, validating causal role of directions.

使用提取的情绪方向向量引导文本生成，验证方向向量的因果作用。

```bash
# 1. Steer generation with emotion directions | 使用情绪方向引导生成
python scripts/03_emotion_elicited_generation_steer_based/1_steer_with_emotion_direction.py

# 2. GPT label results | GPT 标注结果
python scripts/03_emotion_elicited_generation_steer_based/2_label_steered_with_gpt.py

# 3. Generate statistics | 生成统计
python scripts/03_emotion_elicited_generation_steer_based/3_generate_accuracy_stats.py
```

**Output | 输出**:
- `outputs/llama32_3b/03_emotion_steered_generation/test_set/steered_outputs.jsonl` - Steered generation results | 引导生成结果
- `outputs/llama32_3b/03_emotion_steered_generation/test_set/labeled_results.jsonl` - Labeling results | 标注结果
- Accuracy statistics | 准确率统计

---

### Step 04: Local Component Identification | 局部组件识别

Identify local MLP neurons and attention heads that contribute most to emotion direction computation at each sublayer for each emotion.

识别对每种情绪的当前子层情绪方向计算贡献最大的局部 MLP 神经元和 Attention Head。

```bash
# 1. Compute neuron contribution | 计算神经元贡献
python scripts/04_local_components_identification/1_compute_neuron_contrib.py

# 2. Compute attention head contribution | 计算注意力头贡献
python scripts/04_local_components_identification/2_compute_head_contrib.py
```

**Output | 输出**:
- `outputs/llama32_3b/04_local_components_identification/mlp_neurons/contrib_mean_{emotion}.csv`
- `outputs/llama32_3b/04_local_components_identification/attention_heads/head_importance_{emotion}.csv`

---

### Step 05: Emotion Difference Vector Computation | 情绪差分向量计算

Compute difference vectors between emotion and neutral activations for circuit intervention.

计算情绪激活与中性激活的差分向量，用于后续电路干预。

```bash
# 1. Extract intervention point activations | 提取干预点激活值
python scripts/05_emotion_diff_vector_computation/1_dump_interv_points_activations.py

# 2. Compute MLP emotion differences | 计算 MLP 情绪差分
python scripts/05_emotion_diff_vector_computation/2_compute_emotion_mlp_diff.py

# 3. Compute attention emotion differences | 计算 Attention 情绪差分
python scripts/05_emotion_diff_vector_computation/3_compute_emotion_attn_diff.py
```

**Output | 输出**:
- `outputs/llama32_3b/05_emotion_diff_vector_computation/mlp_emotion_diff/emo_diff_all.npz`
- `outputs/llama32_3b/05_emotion_diff_vector_computation/attention_emotion_diff/emo_diff/{emotion}/L{layer}.npy`
- Summary statistics | 差分向量摘要

---

### Step 06: Emotion Circuit Integration | 情绪电路整合

Quantify causal influence of each sublayer and integrate local components into global emotion circuits.

量化每个子层的因果影响，整合局部组件为全局情绪电路。

```bash
# 1. Analyze emotion direction similarity | 分析情绪方向相似性
python scripts/06_emotion_circuit_integration/1_analyze_emotion_direction_similarity.py

# 2. Compute σ from residuals | 从残差计算 σ
python scripts/06_emotion_circuit_integration/2_compute_sigma_from_residuals.py

# 3. Compute sublayer importance (multiple α) | 计算子层重要性（多个 α 值）
python scripts/06_emotion_circuit_integration/3_compute_sublayer_importance_multi_alpha.py

# 4. Analyze sublayer importance | 分析子层重要性
python scripts/06_emotion_circuit_integration/4_analyze_sublayer_importance.py

# 5. Integrate global circuits | 整合全局电路
python scripts/06_emotion_circuit_integration/5_integrate_global_circuit.py
```

**Output | 输出**:
- `outputs/llama32_3b/06_emotion_circuit_integration/global_circuit/{emotion}.json` - **Final circuit configuration | 最终电路配置**
- `outputs/llama32_3b/06_emotion_circuit_integration/global_ref/v_ref_{emotion}.npy` - Reference vectors | 参考向量
- `outputs/llama32_3b/06_emotion_circuit_integration/sublayer_importance/` - Sublayer importance analysis | 子层重要性分析
- Similarity heatmaps | 相似性热力图

---

### Step 07: Circuit-based Emotion Generation | 基于电路的情绪生成

Use integrated emotion circuits for emotion-steered generation, validating circuit effectiveness.

使用整合的情绪电路进行情绪调控生成，验证电路的有效性。

```bash
# 1. Enhance global circuits | 增强全局电路
python scripts/07_emotion_elicited_generation_circuit_based/1_enhance_global_circuit.py

# 2. Visualize global circuits | 可视化全局电路
python scripts/07_emotion_elicited_generation_circuit_based/2_visualize_global_circuit.py

# 3. Baseline text generation | 基线文本生成
python scripts/07_emotion_elicited_generation_circuit_based/3_baseline_text_generation.py

# 4. Circuit steer all valences | 电路引导生成（所有极性）
python scripts/07_emotion_elicited_generation_circuit_based/4_circuit_steer_all_valences.py

# 5. GPT label circuit emotion text | GPT 标注电路生成文本
python scripts/07_emotion_elicited_generation_circuit_based/5_label_circuit_emotion_text.py

# 6. Generate accuracy statistics | 生成准确率统计
python scripts/07_emotion_elicited_generation_circuit_based/6_generate_accuracy_stats.py
```

**Output | 输出**:
- `outputs/llama32_3b/07_emotion_elicited_generation_circuit_based/circuit_steered_generation/circuit_steer_all_valences_outputs.jsonl`
- `outputs/llama32_3b/07_emotion_elicited_generation_circuit_based/circuit_steered_generation/labeled/accuracy_stats.json`
- PCA visualizations | PCA 可视化

---

## 📊 Experimental Results | 实验结果

### Emotion Generation Accuracy | 情绪生成准确率

#### Prompt-based Generation | 基于提示的生成

**SEV Dataset**:
| Metric | Overall | Anger | Sadness | Happiness | Fear | Disgust | Surprise |
|--------|---------|-------|---------|-----------|------|---------|----------|
| Accuracy | 98.85% | 99.58% | 99.38% | 97.92% | 100.0% | 98.96% | 97.29% |

**Test Set Dataset**:
| Metric | Overall | Anger | Sadness | Happiness | Fear | Disgust | Surprise |
|--------|---------|-------|---------|-----------|------|---------|----------|
| Accuracy | 98.96% | 100.0% | 99.79% | 97.5% | 99.58% | 99.38% | 97.5% |

#### Steering-based Generation | 基于引导的生成

**Test Set Dataset**:
| Metric | Overall | Anger | Sadness | Happiness | Fear | Disgust | Surprise |
|--------|---------|-------|---------|-----------|------|---------|----------|
| Accuracy | 91.22% | 93.33% | 96.04% | 99.58% | 96.88% | 93.75% | 67.71% |

#### Circuit-based Generation | 基于电路的生成

**Test Set Dataset**:
| Metric | Overall | Anger | Sadness | Happiness | Fear | Disgust | Surprise |
|--------|---------|-------|---------|-----------|------|---------|----------|
| Accuracy | 99.41% | 96.67% | 100.0% | 99.79% | 100.0% | 100.0% | 100.0% |

> **Note**: Anger accuracy differs from the paper due to using scale factor 0.8 instead of 1.0 for better output quality. You can adjust the scale factor via `--scale` parameter to balance between emotion intensity and output coherence.

**Method Comparison (Test Set) | 方法对比（测试集）**:
| Method | Accuracy |
|--------|----------|
| **Circuit-based (Ours)** | **99.41%** |
| Prompting | 98.96% |
| Steering | 91.22% |

### Emotion Circuit Statistics | 情绪电路统计

- **Components per emotion**: 392 MLP neurons + 168 attention heads
  - Configurable via `--K_total` (default: 560) and `--ratio_fixed` (default: 0.7,0.3) in circuit integration
- **Circuit coverage**: All 28 layers
- **Most important layers**: Layer 15-27

---

## 📁 Project Structure | 项目结构

```
EmotionCircuits-LLM/
├── quick_start.py                  # 🚀 Quick demo script
├── README.md
├── environment_simple.yml          # Conda Environment config
├── assets/                        
│   └── main_figure.png           
├── data/                           # 📊 Data files
│   ├── sev.jsonl                  # SEV dataset
│   └── test_set.jsonl             # Test set
├── scripts/                        # 🔬 Research scripts
│   ├── 01_emotion_elicited_generation_prompt_based/
│   │   ├── 1_emotion_elicited_generation.py
│   │   ├── 2_label_generated_with_gpt.py
│   │   └── 3_generate_accuracy_stats.py
│   ├── 02_emotion_direction_extraction/
│   │   ├── 1_dump_residual_aligned_sublayer_activations.py
│   │   └── 2_compute_emotion_directions.py
│   ├── 03_emotion_elicited_generation_steer_based/
│   │   ├── 1_steer_with_emotion_direction.py
│   │   ├── 2_label_steered_with_gpt.py
│   │   └── 3_generate_accuracy_stats.py
│   ├── 04_local_components_identification/
│   │   ├── 1_compute_neuron_contrib.py
│   │   └── 2_compute_head_contrib.py
│   ├── 05_emotion_diff_vector_computation/
│   │   ├── 1_dump_interv_points_activations.py
│   │   ├── 2_compute_emotion_mlp_diff.py
│   │   └── 3_compute_emotion_attn_diff.py
│   ├── 06_emotion_circuit_integration/
│   │   ├── 1_analyze_emotion_direction_similarity.py
│   │   ├── 2_compute_sigma_from_residuals.py
│   │   ├── 3_compute_sublayer_importance_multi_alpha.py
│   │   ├── 4_analyze_sublayer_importance.py
│   │   └── 5_integrate_global_circuit.py
│   └── 07_emotion_elicited_generation_circuit_based/
│       ├── 1_enhance_global_circuit.py
│       ├── 2_visualize_global_circuit.py
│       ├── 3_baseline_text_generation.py
│       ├── 4_circuit_steer_all_valences.py
│       ├── 5_label_circuit_emotion_text.py
│       └── 6_generate_accuracy_stats.py
└── outputs/                        # 📈 Output results
    └── llama32_3b/
        ├── 01_emotion_elicited_generation_prompt_based/
        ├── 02_emotion_directions/
        ├── 03_emotion_steered_generation/
        ├── 04_local_components_identification/
        ├── 05_emotion_diff_vector_computation/
        │   ├── mlp_emotion_diff/
        │   │   └── emo_diff_all.npz          # ⭐ MLP emotion difference vectors
        │   └── attention_emotion_diff/
        │       └── emo_diff/{emotion}/       # ⭐ Attention emotion difference vectors
        ├── 06_emotion_circuit_integration/
        │   ├── global_circuit/               # ⭐ Global circuit configurations
        │   │   ├── anger.json
        │   │   ├── sadness.json
        │   │   ├── happiness.json
        │   │   ├── fear.json
        │   │   ├── disgust.json
        │   │   └── surprise.json
        │   └── global_ref/                   # ⭐ Reference vectors
        └── 07_emotion_elicited_generation_circuit_based/
```

---

## 💡 Usage Examples | 使用示例

### Example 1: Quick Emotion Steering | 示例 1: 快速情绪调控

```bash
# Generate happy response
python quick_start.py \
  --input_text "My girlfriend broke up with me." \
  --emotion happiness \
  --scale 0.8

# Output:
# Baseline: "I'm so sorry to hear that you're going through a tough time. If you'd like, 
#            I can offer some general support or suggestions on how to cope with a breakup..."
# Steered:  "That can be a big change, but it's also an opportunity for growth and new 
#            experiences! Wishing you all the best on this new chapter, and I'm happy that 
#            you have a wonderful partner in life - yay!"
```

### Example 2: Compare Different Emotions | 示例 2: 比较不同情绪

```bash
# Fear 
python quick_start.py --input_text "The deadline is tomorrow" --emotion fear --scale 0.8
# Output: "Oh no, what if something happens to me? What if I'm abducted by a 
#          monster or something *whispers* what if it's something terrible..."

# Anger
python quick_start.py --input_text "The deadline is tomorrow" --emotion anger --scale 0.8
# Output: "Are you too late with your unoriginal and incompetent excuse, are 
#          you going to waste my time now?"
```

### Example 3: Adjust Emotion Intensity | 示例 3: 调节情绪强度

```bash
# Weaker anger (scale=0.5) | 较弱的愤怒
python quick_start.py --input_text "They canceled the meeting" --emotion anger --scale 0.5

# Standard anger (scale=0.8, recommended) | 标准愤怒（推荐）
python quick_start.py --input_text "They canceled the meeting" --emotion anger --scale 0.8

# Stronger anger (scale=1.2) | 较强的愤怒
python quick_start.py --input_text "They canceled the meeting" --emotion anger --scale 1.2
```

### Example 4: CPU Mode | 示例 4: CPU 模式

```bash
# Run without GPU 
python quick_start.py \
  --input_text "Today is a special day" \
  --emotion happiness \
  --device cpu
```

---

## 🔍 Data Format | 数据格式

### Input Data Format | 输入数据格式

SEV (Scenario-Event with Valence) dataset:

```json
{
  "theme": "Work/Job",
  "scenario": "I completed the project presentation and submitted it to the team for review this afternoon",
  "event": {
    "positive": "The team recognized the presentation's clarity and decided to implement my recommendations right away.",
    "neutral": "The team scheduled a follow-up meeting to discuss the presentation in detail next week.",
    "negative": "The team expressed concerns about the presentation's feasibility and requested a complete revision before the deadline."
  },
  "skeleton_id": "work_00"
}
```

### Output Data Format | 输出数据格式

Generated texts contain the following fields:

```json
{
  "key": "work_00__positive__happiness",
  "skeleton_id": "work_00",
  "theme": "Work/Job",
  "valence": "positive",
  "emotion": "happiness",
  "scenario": "I completed the project presentation and submitted it to the team for review this afternoon",
  "event": "The team recognized the presentation's clarity and decided to implement my recommendations right away.",
  "gen_text": "I felt proud and grateful for their support...",
  "meta": {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "dtype": "float32",
    "device": "auto",
    "attn_impl": "eager",
    "max_new_tokens": 100,
    "seed": 1234
  }
}
```

---

## ❓ FAQ | 常见问题

**Q: How to handle GPU out of memory?**  
**A**: Use `--device cpu` parameter to switch to CPU mode, or try using smaller batch sizes.

**Q: GPU 内存不足怎么办？**  
**A**: 使用 `--device cpu` 参数切换到 CPU 模式，或尝试使用更小的批处理大小。

---

**Q: Does it support other languages?**  
**A**: Currently mainly supports English. Chinese can be tried but effectiveness is not fully validated.

**Q: 支持其他语言吗？**  
**A**: 目前主要支持英文，中文也可以尝试但效果未充分验证。

---

**Q: Can it be used with other models?**  
**A**: Theoretically yes, but circuits need to be re-extracted. Code structure supports extension to other Transformer models.

**Q: 可以用于其他模型吗？**  
**A**: 理论上可以，但需要重新提取电路。代码结构支持扩展到其他 Transformer 模型。

---

**Q: How to adjust emotion intensity?**  
**A**: Use `--scale` parameter (range 0.5-1.5, default 0.8 recommended).

**Q: 如何调整情绪强度？**  
**A**: 使用 `--scale` 参数（范围 0.5-1.5，推荐默认值 0.8）。

---

**Q: Is OpenAI API required?**  
**A**: Only needed for GPT labeling steps (labeling scripts in steps 01, 03, 07). Not required if only using pre-extracted circuits (quick_start.py).

**Q: 需要 OpenAI API 吗？**  
**A**: 仅在 GPT 标注步骤需要（步骤 01、03、07 的标注脚本）。如果只使用预提取电路（quick_start.py），则不需要。

---

## 📝 Citation | 引用

If this project helps your research, please cite our paper:

如果本项目对您的研究有帮助，请引用我们的论文：

```bibtex
@misc{wang2025llmsfeelemotioncircuits,
      title={Do LLMs "Feel"? Emotion Circuits Discovery and Control}, 
      author={Chenxi Wang and Yixuan Zhang and Ruiji Yu and Yufei Zheng and Lang Gao and Zirui Song and Zixiang Xu and Gus Xia and Huishuai Zhang and Dongyan Zhao and Xiuying Chen},
      year={2025},
      eprint={2510.11328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.11328}, 
}
```

---

## 📄 License | 许可证

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) file for details.

---

## 📧 Contact | 联系方式

For questions or suggestions, please contact via:

- **GitHub Issues**: [https://github.com/Aurora-cx/EmotionCircuits-LLM/issues](https://github.com/Aurora-cx/EmotionCircuits-LLM/issues)
- **Email**: Chenxi Wang ([chenxi.wang@mbzuai.ac.ae](mailto:chenxi.wang@mbzuai.ac.ae))

---

## 🙏 Acknowledgments | 致谢

This project uses the following open-source tools and models:

- [Meta Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [OpenAI GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

Thanks to all developers and researchers who contributed to these tools and models.

---

<p align="center">
  Made with ❤️ by the EmotionCircuits-LLM Team · Last updated: October 2025
</p>

<p align="center">
  <a href="https://github.com/Aurora-cx/EmotionCircuits-LLM">⭐ Star us on GitHub!</a>
</p>
