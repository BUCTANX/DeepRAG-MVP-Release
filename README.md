# 🧠 DeepRAG: 你的专属垂直领域 AI 助手构建器


> 上传 PDF -> 自动微调 -> 拥有一个懂你业务的专属 AI

DeepRAG 是一个“开箱即用”的工具，它能帮你把私有的 PDF 文档（如医疗指南、法律条文、员工手册）变成一个聪明的 AI 对话助手。

![Python](https://img.shields.io/badge/Python-3.10-blue)
![GPU](https://img.shields.io/badge/GPU-RTX3060%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖥️ 我能用它做什么？

1.  **上传文档**：比如上传一份《糖尿病饮食指南.pdf》。
2.  **一键学习**：点击“自动化微调”，AI 会自己阅读文档并进行自我训练（LoRA 微调）。
3.  **精准问答**：训练好后，你问它“糖尿病人能吃西瓜吗？”，它会根据文档精准回答，而不是胡说八道。

---

## 🚀 极速部署 (3分钟上手)

### 前置要求
*   **电脑配置**：需要一张 NVIDIA 显卡（显存 6GB 以上，推荐 RTX 3060 及以上）。
*   **系统**：推荐使用 **Windows + WSL2** 或 **Linux**。
*   **基础软件**：请确保已安装 [Miniconda](https://docs.anaconda.com/miniconda/) 或 Anaconda。

### 步骤 1: 下载代码
点击右上角的绿色按钮 **Code -> Download ZIP**，解压到一个文件夹。
或者使用 Git：
```bash
git clone https://github.com/BUCTANX/DeepRAG-MVP-Release.git
cd DeepRAG-MVP-Release
步骤 2: 一键启动 (推荐 🔥)
我们提供了一个自动化脚本，帮你完成所有复杂的环境配置。

在终端（Terminal）中运行：

Bash

chmod +x start.sh  # 赋予执行权限 (仅第一次需要)
./start.sh
脚本会自动执行以下操作：

创建名为 deeprag_env 的虚拟环境。
自动安装 PyTorch 和所有 AI 依赖库。
启动网页界面。
步骤 3: 开始使用
当终端显示 Network URL: http://localhost:8501 时，打开浏览器访问该地址即可。

📖 手动安装指南 (如果你不想用脚本)
如果你是开发者，也可以手动一步步安装：

创建环境

Bash

conda create -n deeprag python=3.10
conda activate deeprag
安装 PyTorch (关键)

Bash

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
安装依赖

Bash

pip install -r requirements.txt
运行

Bash

streamlit run app.py --server.address=localhost
❓ 常见问题 (Q&A)
Q: 第一次运行为什么这么慢？
A: 首次运行时，程序需要从 HuggingFace 下载 Qwen2-1.5B 模型（约 3GB）和 Embedding 模型。请耐心等待，保持网络通畅。

Q: 显存不足 (OOM) 怎么办？
A: 本项目默认开启 4-bit 量化，最低支持 6GB 显存。如果依然报错，请尝试关闭其他占用显存的程序。

Q: Windows 用户可以直接运行吗？
A: 由于核心依赖库 bitsandbytes 对 Windows 支持不佳，强烈建议在 WSL2 (Ubuntu) 下运行。在原生 Windows PowerShell 下运行可能会报错。

📂 目录结构说明
text

DeepRAG/
├── app.py                 # 网页主程序 (Streamlit)
├── start.sh               # 一键启动脚本
├── core/                  # AI 核心算法代码
│   ├── trainer.py         # 负责模型微调
│   └── rag_engine.py      # 负责问答推理
├── data/                  # 存放你的数据
└── models/                # 存放训练好的模型
