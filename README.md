# 🧠 DeepRAG：基于 RAG + LoRA 的私有领域大模型构建平台

DeepRAG 是一个**端到端**的垂直领域大模型定制工具。  
用户只需上传私有文档（PDF），即可自动完成数据清洗、指令微调（LoRA），最终得到一个同时具备**领域知识（RAG）**与**领域适应性（Fine-tuning）**的专属智能助手。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)
![GPU](https://img.shields.io/badge/GPU-RTX3060%2B-orange)

## ✨ 核心特性

- **全自动微调流水线**：上传 PDF → 自动生成高质量 QA 数据 → 一键 LoRA 微调
- **极低资源需求**：4-bit 量化训练，RTX 3060 (6GB) 即可微调 Qwen2-1.5B
- **RAG + Fine-tuning 双引擎**：大幅降低幻觉率，提升领域问题回答精准度
- **所见即所得 Web 界面**：基于 Streamlit，操作简单直观

## 🛠️ 快速开始

### 1. 环境要求

- 操作系统：Linux（推荐 Ubuntu 20.04+ 或 WSL2）
- 显卡：NVIDIA GPU（显存 ≥ 6GB）
- Python：3.10+
- CUDA：11.8+

### 2. 安装依赖（推荐使用 Conda）

```bash
# 创建并激活虚拟环境
conda create -n deeprag python=3.10 -y
conda activate deeprag

# 安装 PyTorch（以 CUDA 11.8 为例，根据实际情况调整）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
3. 启动项目
Bashstreamlit run app.py --server.address=0.0.0.0
# 或仅本地访问
# streamlit run app.py
启动后，浏览器访问显示的地址（默认：http://localhost:8501）
📖 使用流程

上传领域文档
在左侧边栏上传你的 PDF 文件（如员工手册、医疗指南、产品说明书等）
一键启动自动化微调
点击「开始自动化微调」按钮，系统将自动完成：
PDF 解析 → 文本清洗 → QA 数据生成 → LoRA 微调首次运行会自动下载 Qwen2-1.5B-Instruct 模型，请保持网络畅通
开始智能对话
训练完成后，在右侧对话框直接提问，系统会同时使用：
微调后的 LoRA 适配器
实时 RAG 检索你的文档内容


🗂 项目目录结构
textDeepRAG/
├── app.py                 # Streamlit WebUI 主程序
├── core/
│   ├── data_processor.py  # PDF解析、清洗与指令数据生成
│   ├── trainer.py         # LoRA 微调逻辑（基于 TRL）
│   └── rag_engine.py      # RAG 检索与生成引擎
├── data/                  # 上传的原始文档 & 处理后的数据
├── models/                # LoRA 适配器保存目录
├── requirements.txt
└── README.md
⚠️ 重要注意事项

默认基座模型：Qwen/Qwen2-1.5B-Instruct
微调显存占用：约 4–5GB（4-bit 量化）
Windows 用户强烈建议使用 WSL2
原生 Windows 可能遇到 bitsandbytes 等兼容性问题
建议在训练前关闭其他占用显存的程序
