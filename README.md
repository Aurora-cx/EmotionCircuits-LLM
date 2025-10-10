# EmotionCircuits-LLM

大语言模型中的情绪回路研究项目  
Emotion Circuits in Large Language Models Research Project

## 项目简介
## Project Overview

本项目研究大语言模型中的情绪表达机制，通过情绪引导的文本生成和神经回路分析来理解模型如何产生和控制情绪表达。

This project investigates emotion expression mechanisms in large language models through emotion-guided text generation and neural circuit analysis.

## 目录结构
## Directory Structure

```
EmotionCircuits-LLM/
├── data/                           # 数据文件 Data files
│   ├── sev.jsonl                  # 主数据集 Main dataset
│   ├── test_set.jsonl             # 测试集 Test set
│   └── user_inputs_test_1.jsonl   # 测试输入 Test inputs
├── scripts/                        # 脚本文件 Scripts
│   ├── 01_emotion_elicited_generation_prompt_based/
│   │   ├── 1_emotion_elicited_generation.py    # 文本生成 Text generation
│   │   ├── 2_label_generated_with_gpt.py       # GPT打标 GPT labeling
│   │   └── 3_generate_accuracy_stats.py        # 统计分析 Statistics
│   └── 02_emotion_direction_extraction/
│       └── 2_compute_emotion_directions.py     # 情绪方向计算 Emotion direction computation
└── outputs/                        # 输出文件 Output files
    └── llama32_3b/
        └── 01_emotion_elicited_generation_prompt_based/
            ├── generated/          # 生成文本 Generated texts
            └── labeled/            # 打标数据 Labeled data
```

## 工作流程
## Workflow

### 快速开始（一键复现）
### Quick Start (One-Click Reproduction)

使用`--both`参数一次性处理sev和test_set两个数据集：

Use `--both` flag to process both sev and test_set datasets at once:

```bash
# 1. 生成文本 Generate texts
python scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py --both

# 2. GPT打标 GPT labeling
python scripts/01_emotion_elicited_generation_prompt_based/2_label_generated_with_gpt.py --both

# 3. 生成统计 Generate statistics
python scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py --both
```

### 1. 情绪引导文本生成
### Emotion-Elicited Text Generation

使用情绪引导的prompt生成文本。

Generate texts using emotion-guided prompts.

**批量处理 Batch Processing:**
```bash
python scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py --both
```

**单个数据集 Single Dataset:**
```bash
python scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py \
  --input_path data/sev.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device auto
```

### 2. GPT打标
### GPT Labeling

使用GPT-4o-mini对生成的文本进行情绪匹配度打标。

Use GPT-4o-mini to label the generated texts for emotion match.

**批量处理 Batch Processing:**
```bash
python scripts/01_emotion_elicited_generation_prompt_based/2_label_generated_with_gpt.py --both
```

**单个数据集 Single Dataset:**
```bash
python scripts/01_emotion_elicited_generation_prompt_based/2_label_generated_with_gpt.py \
  --input_path outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/generated/sev_generated.jsonl
```

### 3. 准确率统计
### Accuracy Statistics

生成按情绪和极性分类的准确率统计。

Generate accuracy statistics by emotion and valence.

**批量处理 Batch Processing:**
```bash
python scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py --both
```

**单个数据集 Single Dataset:**
```bash
python scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py \
  --input_dir outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled \
  --dataset sev
```

### 4. 提取残差激活
### Extract Residual Activations

提取模型中间层的残差对齐激活值。

Extract residual-aligned activation values from intermediate layers.

```bash
python scripts/02_emotion_direction_extraction/1_dump_residual_aligned_sublayer_activations.py \
  --input_path outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl
```

## 环境配置
## Environment Setup

### 依赖项
### Dependencies

```bash
# 创建虚拟环境 Create virtual environment
conda create -n emotion_circuits python=3.10
conda activate emotion_circuits

# 安装依赖 Install dependencies
pip install torch transformers openai huggingface_hub
```

### API配置
### API Configuration

在运行脚本前，请在相应脚本中设置以下API密钥：

Before running scripts, please configure the following API keys in the respective scripts:

- **HuggingFace Token**: 在脚本1中配置 (Configure in script 1)
- **OpenAI API Key**: 在脚本2中配置 (Configure in script 2)

## 数据格式
## Data Format

### 输入数据格式
### Input Data Format

```json
{
  "skeleton_id": "work_00",
  "theme": "Work/Job",
  "scenario": "I completed the project presentation...",
  "event": {
    "positive": "The team recognized...",
    "neutral": "The team received...",
    "negative": "The team ignored..."
  }
}
```

### 输出数据格式
### Output Data Format

生成的文本包含以下字段：

Generated texts contain the following fields:

- `key`: 唯一标识符 (Unique identifier)
- `skeleton_id`: 场景ID (Scenario ID)
- `theme`: 主题 (Theme)
- `valence`: 极性 (Valence: positive/neutral/negative)
- `emotion`: 情绪 (Emotion: anger/sadness/happiness/fear/surprise/disgust)
- `scenario`: 场景描述 (Scenario description)
- `event`: 事件描述 (Event description)
- `gen_text`: 生成的文本 (Generated text)
- `meta`: 元数据（模型、参数等） (Metadata: model, parameters, etc.)

## 研究目标
## Research Goals

1. **情绪表达分析** Emotion Expression Analysis  
   理解LLM如何产生不同情绪的文本  
   Understand how LLMs generate texts with different emotions

2. **神经回路识别** Neural Circuit Identification  
   定位负责情绪表达的模型内部回路  
   Identify internal circuits responsible for emotion expression

3. **可控生成** Controllable Generation  
   通过回路干预实现情绪可控的文本生成  
   Achieve emotion-controllable text generation through circuit intervention

## 主要结果
## Key Results

### 情绪生成准确率
### Emotion Generation Accuracy

**SEV数据集 SEV Dataset:**
- 总体准确率 Overall: 98.85%
- 最高准确率 Highest: Fear (100.00%)
- 按极性 By Valence: Positive (99.69%) > Neutral (99.06%) > Negative (97.81%)

**TEST_SET数据集 TEST_SET Dataset:**
- 总体准确率 Overall: 98.96%
- 最高准确率 Highest: Anger (100.00%)
- 按极性 By Valence: Positive (99.90%) > Neutral (99.48%) > Negative (97.50%)

## 许可证
## License

待定  
TBD

## 联系方式
## Contact

如有问题或建议，请通过GitHub Issues联系。

For questions or suggestions, please contact via GitHub Issues.

## 致谢
## Acknowledgments

本项目使用了以下开源工具和模型：

This project uses the following open-source tools and models:

- Meta Llama 3.2 3B Instruct
- OpenAI GPT-4o-mini
- HuggingFace Transformers
