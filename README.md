# 🧠 DeepRAG: 基于 RAG + LoRA 的私有领域大模型构建平台

DeepRAG 是一个端到端的垂直领域大模型构建工具。它允许用户上传私有文档（PDF），自动完成数据清洗、指令微调（LoRA），并生成一个既具备领域知识（RAG）又具备领域适应性（Fine-tuning）的专属智能助手。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![GPU](https://img.shields.io/badge/GPU-RTX3060%2B-orange)

## ✨ 核心特性

- **自动化微调流水线**：上传 PDF -> 自动生成 QA 数据 -> 一键启动 LoRA 微调。
- **低资源消耗**：支持 4-bit 量化训练，RTX 3060 (6GB) 即可运行 Qwen2-1.5B 微调。
- **RAG + Fine-tuning 双引擎**：结合检索增强生成与参数微调，大幅降低幻觉，提升回答精准度。
- **所见即所得**：基于 Streamlit 的 WebUI，交互简单直观。

## 🛠️ 快速开始

### 1. 环境要求
- 操作系统：Linux (推荐 Ubuntu 20.04+ 或 WSL2)
- 显卡：NVIDIA GPU (显存 >= 6GB)
- Python：3.10+
- CUDA：11.8+

### 2. 安装依赖

强烈建议使用 Conda 创建独立环境：

bash
conda create -n deeprag python=3.10
conda activate deeprag

## 安装 PyTorch (根据你的 CUDA 版本调整，这里以 11.8 为例)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
### 3. 运行项目
Bash

streamlit run app.py --server.address=localhost

启动后，访问浏览器显示的地址（通常是 http://localhost:8501）。

📖 使用流程
准备数据：在左侧边栏上传你的 PDF 文档（如《员工手册》、《医疗指南》）。
一键微调：点击“启动自动化微调”，系统会自动解析 PDF，生成训练数据，并开始训练。
注：首次运行会自动下载 Qwen2-1.5B 模型，请保持网络通畅（代码内置 HF 镜像加速）。
智能问答：微调完成后，在右侧对话框提问。系统会加载你刚刚训练好的 LoRA 适配器，并结合文档内容回答。
🏗️ 目录结构
text

DeepRAG/
├── app.py                 # WebUI 入口
├── core/                  # 核心逻辑
│   ├── data_processor.py  # 数据清洗与生成
│   ├── trainer.py         # LoRA 微调脚本 (基于 TRL)
│   └── rag_engine.py      # RAG 推理引擎
├── data/                  # 数据存储目录
└── models/                # 模型权重存储目录
⚠️ 注意事项
本项目默认使用 Qwen/Qwen2-1.5B-Instruct 作为基座模型。
微调过程中显存占用约为 4GB-5GB。
Windows 用户请务必使用 WSL2 运行，原生 Windows 可能存在 bitsandbytes兼容性问题。
