🧠 DeepRAG：你的专属垂直领域 AI 助手构建器

上传 PDF → 自动微调 → 拥有一个真正懂你业务的专属 AI

DeepRAG 是一个 开箱即用的垂直领域 AI 助手构建工具，
它可以将你的 私有 PDF 文档（如医疗指南、法律条文、员工手册）
快速转化为一个 高可信、可对话的专业 AI。






🖥️ 我能用它做什么？

上传文档
上传如《糖尿病饮食指南.pdf》《公司员工手册.pdf》等专业资料。

一键学习
点击 自动化微调，系统将：

自动解析 PDF

构建向量数据库

使用 LoRA 对大模型进行轻量微调

精准问答
训练完成后，你可以提问：

“糖尿病人能吃西瓜吗？”

AI 将 严格基于你的文档内容回答，避免幻觉。

🚀 极速部署（3 分钟上手）
✅ 前置要求

硬件

NVIDIA 显卡（≥ 6GB 显存）

推荐：RTX 3060 / 4060 / 4090

系统

✅ Linux

✅ Windows + WSL2（Ubuntu）

❌ 不推荐原生 Windows

软件

已安装 Miniconda / Anaconda

📥 步骤 1：下载代码
方法一：ZIP 下载

点击右上角 Code → Download ZIP，解压即可。

方法二：Git 克隆（推荐）
git clone https://github.com/BUCTANX/DeepRAG-MVP-Release.git
cd DeepRAG-MVP-Release

⚡ 步骤 2：一键启动（强烈推荐 🔥）

项目提供自动化脚本，帮你完成所有复杂环境配置。

chmod +x start.sh   # 仅首次需要
./start.sh

脚本将自动完成：

创建 Conda 环境：deeprag_env

安装 PyTorch（CUDA 版本）

安装全部依赖

启动 Streamlit Web UI

🌐 步骤 3：开始使用

当终端出现以下提示：

Network URL: http://localhost:8501


在浏览器中打开该地址即可 🎉

📖 手动安装指南（开发者向）

适合希望自定义环境或调试源码的用户

1️⃣ 创建环境
conda create -n deeprag python=3.10
conda activate deeprag

2️⃣ 安装 PyTorch（⚠️ 关键）
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

3️⃣ 安装依赖
pip install -r requirements.txt

4️⃣ 启动应用
streamlit run app.py --server.address=localhost

❓ 常见问题（Q&A）
Q1：第一次运行为什么很慢？

A：
首次运行需要从 HuggingFace 下载模型：

Qwen2-1.5B（≈ 3GB）

Embedding 模型

请确保网络通畅，耐心等待。

Q2：显存不足（OOM）怎么办？

A：

项目默认使用 4-bit 量化

最低支持 6GB 显存

建议：

关闭其他占用显存的软件

不要同时运行多个模型

Q3：Windows 可以直接运行吗？

A：不推荐。

原因：

bitsandbytes 对 Windows 原生支持不稳定

容易出现 CUDA / 编译错误

✅ 强烈推荐使用 WSL2（Ubuntu）

📂 项目目录结构
DeepRAG/
├── app.py                 # Streamlit Web 主程序
├── start.sh               # 一键启动脚本
├── core/                  # AI 核心逻辑
│   ├── trainer.py         # LoRA 微调逻辑
│   └── rag_engine.py      # RAG 问答引擎
├── data/                  # 用户上传的 PDF 数据
└── models/                # 训练完成的模型
