# EmotionCircuits-LLM

情绪回路大语言模型研究项目  
Emotion Circuits in Large Language Models Research Project

## 项目简介 | Project Overview

本项目研究大语言模型中的情绪表达机制，通过情绪引导的文本生成和神经回路分析来理解模型如何产生和控制情绪表达。

This project investigates emotion expression mechanisms in large language models through emotion-guided text generation and neural circuit analysis.

## 目录结构 | Directory Structure

```
EmotionCircuits-LLM/
├── data/                           # 数据文件 | Data files
│   ├── sev.jsonl                  # 主数据集 | Main dataset
│   └── test_set.jsonl             # 测试集 | Test set
├── scripts/                        # 脚本文件 | Scripts
│   ├── 01_emotion_elicited_generation_prompt_based/  # 情绪引导生成 | Emotion-elicited generation
│   │   ├── 1_emotion_elicited_generation.py          # 文本生成 | Text generation
│   │   ├── 2_label_generated_with_gpt.py             # GPT打标 | GPT labeling
│   │   └── 3_generate_accuracy_stats.py              # 准确率统计 | Accuracy statistics
│   └── 02_emotion_direction_extraction/              # 情绪方向提取 | Emotion direction extraction
│       └── 1_dump_residual_aligned_sublayer_activations.py
└── outputs/                        # 输出文件 | Output files (gitignored)
```

## 工作流程 | Workflow

### 1. 情绪引导文本生成 | Emotion-Elicited Text Generation

使用情绪引导的prompt生成文本：

```bash
python scripts/01_emotion_elicited_generation_prompt_based/1_emotion_elicited_generation.py \
  --input_path data/sev.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device auto
```

### 2. GPT打标 | GPT Labeling

使用GPT-4o-mini对生成的文本进行情绪匹配度打标：

```bash
python scripts/01_emotion_elicited_generation_prompt_based/2_label_generated_with_gpt.py \
  --input_path outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/generated/sev_generated.jsonl
```

### 3. 准确率统计 | Accuracy Statistics

生成按情绪和极性分类的准确率统计：

```bash
python scripts/01_emotion_elicited_generation_prompt_based/3_generate_accuracy_stats.py \
  --input_dir outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled
```

### 4. 提取残差激活 | Extract Residual Activations

提取模型中间层的残差对齐激活值：

```bash
python scripts/02_emotion_direction_extraction/1_dump_residual_aligned_sublayer_activations.py \
  --input_path outputs/llama32_3b/01_emotion_elicited_generation_prompt_based/labeled/sev/accepted.jsonl
```

## 环境配置 | Environment Setup

### 依赖项 | Dependencies

```bash
conda create -n emotion_circuits python=3.10
conda activate emotion_circuits
pip install torch transformers openai huggingface_hub
```

### API配置 | API Configuration

在运行脚本前，请设置以下API密钥：

- **HuggingFace Token**: 在脚本1中配置
- **OpenAI API Key**: 在脚本2中配置

## 数据格式 | Data Format

### 输入数据格式 | Input Data Format

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

### 输出数据格式 | Output Data Format

生成的文本包含以下字段：
- `key`: 唯一标识符
- `skeleton_id`: 场景ID
- `theme`: 主题
- `valence`: 极性 (positive/neutral/negative)
- `emotion`: 情绪 (anger/sadness/happiness/fear/surprise/disgust)
- `scenario`: 场景描述
- `event`: 事件描述
- `gen_text`: 生成的文本
- `meta`: 元数据（模型、参数等）

## 研究目标 | Research Goals

1. **情绪表达分析**: 理解LLM如何产生不同情绪的文本
2. **神经回路识别**: 定位负责情绪表达的模型内部回路
3. **可控生成**: 通过回路干预实现情绪可控的文本生成

## 许可证 | License

待定 | TBD

## 联系方式 | Contact

如有问题或建议，请通过GitHub Issues联系。

For questions or suggestions, please contact via GitHub Issues.

