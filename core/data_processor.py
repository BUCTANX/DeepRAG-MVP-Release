import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf_to_training_data(pdf_path, output_json_path):
    """
    1. 读取 PDF
    2. 切分文本
    3. 构造微调格式 (Instruction Tuning Format)
    """
    print(f"📄 Processing {pdf_path}...")
    
    # 1. 加载
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. 切分 (为了微调，块可以稍微大一点，或者按段落)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    training_data = []
    
    # 3. 构造 QA 对 (这里模拟一种自监督学习：让模型学会复述知识)
    # 在真实产品中，这里会调用 GPT-4 提取 QA
    for split in splits:
        text = split.page_content.strip()
        if len(text) < 50: continue # 跳过太短的
        
        # 构造一条“阅读理解”风格的指令
        item = {
            "instruction": "Please explain the following content in detail.",
            "input": text[:50] + "...", # 提示词取开头
            "output": text # 让模型学会输出这段知识
        }
        training_data.append(item)
        
        # 再构造一条“知识问答”风格 (模拟)
        item2 = {
            "instruction": "What information is provided in the document?",
            "input": "",
            "output": f"The document mentions: {text}"
        }
        training_data.append(item2)
        
    # 保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Generated {len(training_data)} training samples at {output_json_path}")
    return len(training_data)