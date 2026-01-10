import streamlit as st
import os
import shutil

# --- 延迟导入核心模块 ---
# 我们不在这里 import core.xxx，防止启动卡顿
# from core.data_processor import process_pdf_to_training_data
# from core.trainer import train_user_model
# from core.rag_engine import RAGEngine

# 初始化路径
UPLOAD_DIR = "data/uploads"
DATA_DIR = "data/processed"
MODEL_DIR = "models/user_adapters"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="DeepRAG Customizer", layout="wide")
st.title("🧠 DeepRAG: 打造你的专属垂直领域模型")

# --- 侧边栏：上传与微调 ---
with st.sidebar:
    st.header("1. 上传私有文档")
    uploaded_file = st.file_uploader("上传 PDF 资料", type=["pdf"])
    
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已保存: {uploaded_file.name}")
        
        st.header("2. 开始微调")
        if st.button("🚀 启动自动化微调"):
            # --- 关键修改：在这里才 Import ---
            with st.spinner("正在加载 AI 核心引擎 (首次运行较慢)..."):
                from core.data_processor import process_pdf_to_training_data
                from core.trainer import train_user_model
                
            with st.status("正在进行自动化微调...", expanded=True) as status:
                st.write("⚙️ 正在解析 PDF 并生成训练数据...")
                json_path = os.path.join(DATA_DIR, "train.json")
                count = process_pdf_to_training_data(file_path, json_path)
                st.write(f"✅ 生成了 {count} 条训练样本。")
                
                st.write("🏋️‍♂️ 正在调用 GPU 进行 LoRA 微调...")
                adapter_name = uploaded_file.name.split('.')[0]
                output_path = os.path.join(MODEL_DIR, adapter_name)
                
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)
                    
                train_user_model(json_path, output_path)
                st.write("✅ 微调完成！模型已保存。")
                
                st.session_state["current_adapter"] = output_path
                st.session_state["current_pdf"] = file_path
                status.update(label="微调流程结束！", state="complete", expanded=False)

# --- 主界面：问答 ---
st.header("3. 智能问答 (RAG + Fine-tuned)")

if "current_adapter" in st.session_state:
    adapter_path = st.session_state["current_adapter"]
    pdf_path = st.session_state["current_pdf"]
    
    st.info(f"当前使用的微调模型: {os.path.basename(adapter_path)}")
    
    # 这里的 Import 也放到函数里
    @st.cache_resource
    def get_engine(adapter_path, pdf_path):
        from core.rag_engine import RAGEngine # 延迟导入
        engine = RAGEngine()
        engine.load_model(adapter_path)
        engine.build_index(pdf_path)
        return engine
        
    engine = get_engine(adapter_path, pdf_path)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("向你的专属模型提问..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("AI 正在思考..."):
                response = engine.chat(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("请先在左侧上传 PDF 并完成微调。")