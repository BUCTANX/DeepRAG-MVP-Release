## 🧠 DeepRAG：你的专属垂直领域 AI 助手构建器

> **上传 PDF → 自动微调 → 拥有一个真正懂你业务的专属 AI**

DeepRAG 是一个 **开箱即用的垂直领域 AI 助手构建工具**，  
它可以将你的 **私有 PDF 文档**（如医疗指南、法律条文、员工手册）  
快速转化为一个 **高可信、可对话的专业 AI**。

![Python](https://img.shields.io/badge/Python-3.10-blue)  
![GPU](https://img.shields.io/badge/GPU-RTX3060%2B-orange)  
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖥️ 我能用它做什么？

1. **上传文档**  
   上传如《糖尿病饮食指南.pdf》《公司员工手册.pdf》等专业资料。

2. **一键学习**  
   点击 **自动化微调**，系统将：
   - 自动解析 PDF  
   - 构建向量数据库  
   - 使用 **LoRA** 对大模型进行轻量微调  

3. **精准问答**  
   训练完成后，你可以提问：

   > 糖尿病人能吃西瓜吗？

   AI 将 **严格基于你的文档内容回答**，避免幻觉。

---

## 🚀 极速部署（3 分钟上手）

### ✅ 前置要求

- **硬件**
  - NVIDIA 显卡（≥ 6GB 显存）
  - 推荐：RTX 3060 / 4060 / 4090

- **系统**
  - Linux  
  - Windows + WSL2（Ubuntu）

- **软件**
  - 已安装 Miniconda / Anaconda

---

## 📥 步骤 1：下载代码

### 方法一：ZIP 下载

点击右上角 **Code → Download ZIP**，解压即可。

### 方法二：Git 克隆（推荐）

```bash
git clone https://github.com/BUCTANX/DeepRAG-MVP-Release.git
cd DeepRAG-MVP-Release
```

---

## ⚡ 步骤 2：一键启动（推荐）

```bash
chmod +x start.sh
./start.sh
```

脚本会自动完成：

- 创建 Conda 环境 `deeprag_env`
- 安装 PyTorch（CUDA）
- 安装全部依赖
- 启动 Streamlit Web UI

---

## 🌐 步骤 3：开始使用

当终端出现：

```
Network URL: http://localhost:8501
```

在浏览器中打开即可。

---

## 📖 手动安装指南（可选）

### 1. 创建环境

```bash
conda create -n deeprag python=3.10
conda activate deeprag
```

### 2. 安装 PyTorch

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动应用

```bash
streamlit run app.py --server.address=localhost
```

---

## ❓ 常见问题（Q&A）

### Q1：第一次运行为什么很慢？

首次运行需要下载模型（约 3GB），请耐心等待。

### Q2：显存不足怎么办？

- 默认 4-bit 量化  
- 最低 6GB 显存  
- 请关闭其他占用显存的程序  

### Q3：Windows 可以直接运行吗？

不推荐，建议使用 **WSL2（Ubuntu）**。

---

## 📂 项目目录结构

```text
DeepRAG/
├── app.py                 # Streamlit 主程序
├── start.sh               # 一键启动脚本
├── core/
│   ├── trainer.py         # 微调逻辑
│   └── rag_engine.py      # RAG 推理
├── data/
└── models/
```
