# 🧠 DeepRAG：基于 RAG + LoRA 的私有领域大模型构建平台

DeepRAG 是一个**端到端**的垂直领域大模型定制工具。  
用户只需上传私有文档（PDF），即可自动完成数据清洗、指令微调（LoRA），最终得到一个同时具备**领域知识（RAG）**与**领域适应性（Fine-tuning）**的专属智能助手。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![GPU](https://img.shields.io/badge/GPU-RTX3060%2B-orange)

## ✨ 核心特性

- 全自动微调流水线：上传 PDF → 自动生成高质量 QA 数据 → 一键 LoRA 微调
- 极低资源需求：4-bit 量化训练，RTX 3060 (6GB) 即可微调 Qwen2-1.5B
- RAG + Fine-tuning 双引擎：大幅降低幻觉率，提升领域问题回答精准度
- 所见即所得 Web 界面：基于 Streamlit，操作简单直观

## 🛠️ 快速开始

### 1. 环境要求

- 操作系统：Linux（推荐 Ubuntu 20.04+ 或 WSL2）
- 显卡：NVIDIA GPU（显存 ≥ 6GB）
- Python：3.10+
- CUDA：11.8+

### 2. 安装依赖（推荐使用 Conda）

```bash
conda create -n deeprag python=3.10 -y
conda activate deeprag
Bashpip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
Bashpip install -r requirements.txt
3. 启动项目
Bash# 推荐：允许局域网内其他设备访问
streamlit run app.py --server.address=0.0.0.0

# 或者仅本地访问
# streamlit run app.py
启动后浏览器访问：http://localhost:8501（或你机器的内网 IP:8501）
📖 使用流程

在左侧边栏上传你的 PDF 领域文档
点击「开始自动化微调」按钮，耐心等待完成
训练完成后，直接在右侧对话框开始提问

注：首次运行会自动从 HuggingFace 下载 Qwen2-1.5B-Instruct 模型
🗂 项目目录结构
textDeepRAG/
├── app.py                 # Streamlit WebUI 主程序
├── core/
│   ├── data_processor.py  # PDF解析 + 清洗 + 指令数据生成
│   ├── trainer.py         # LoRA 微调（基于 TRL）
│   └── rag_engine.py      # RAG 检索与融合逻辑
├── data/                  # 原始文档 & 处理后的中间数据
├── models/                # 训练好的 LoRA 适配器存放处
├── requirements.txt
└── README.md
⚠️ 重要提醒

默认基座模型：Qwen/Qwen2-1.5B-Instruct
微调显存峰值：约 4.2～5.1GB（4-bit 量化）
Windows 用户强烈建议使用 WSL2
建议训练前关闭其他显存占用程序

祝你快速打造出专属的领域智能助手！ 🚀
