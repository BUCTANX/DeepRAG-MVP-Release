import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

class RAGEngine:
    def __init__(self, base_model_id="Qwen/Qwen2-1.5B-Instruct"):
        self.base_model_id = base_model_id
        self.tokenizer = None
        self.model = None
        self.llm = None
        self.vectorstore = None
        
    def load_model(self, adapter_path=None):
        """
        加载模型 (强制使用 4-bit 量化 + 指定 GPU，防止 Meta Tensor 报错)
        """
        print("🤖 Loading Base Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        
        # --- 关键修改：使用 4-bit 量化配置 ---
        # 这不仅省显存，还能避免 meta device 报错
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # --- 关键修改：device_map={"": "cuda"} ---
        # 强制所有层都在 GPU 上，禁止 accelerate 切分模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map={"": "cuda"}, # 显式指定 GPU，拒绝 auto
            torch_dtype=torch.float16
        )
        
        if adapter_path and os.path.exists(adapter_path):
            print(f"✨ Loading Adapter from {adapter_path}...")
            # 加载 LoRA
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print("⚠️ No adapter found, utilizing Base Model.")
            self.model = base_model
            
        # 构建 pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300, # 稍微调大一点，让它多说点
            temperature=0.7
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ Model Loaded Successfully.")
        
    def build_index(self, pdf_path):
        """
        构建 RAG 索引
        """
        print("📚 Indexing PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = FAISS.from_documents(texts, embeddings)
        print("✅ Index built.")
        
    def chat(self, query):
        if not self.vectorstore:
            return "请先上传文档构建知识库。"
            
        # RAG 检索
        docs = self.vectorstore.similarity_search(query, k=2)
        context = "\n".join([d.page_content for d in docs])
        
        # 构造 Prompt
        template = """Answer the question based on the context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        final_prompt = prompt.format(context=context, question=query)
        
        # 生成
        res = self.llm.invoke(final_prompt)
        
        # 清洗结果 (去掉 prompt 部分)
        # 有时候模型会把 Prompt 复述一遍，这里做一个简单的截断
        if final_prompt in res:
            return res.split("Answer:")[-1].strip()
        return res.strip()