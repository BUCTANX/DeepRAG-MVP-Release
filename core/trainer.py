import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# 确保环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def train_user_model(data_path, output_dir, base_model_id="Qwen/Qwen2-1.5B-Instruct"):
    """
    启动微调流程
    """
    print(f"🚀 Starting training task...")
    print(f"   Data: {data_path}")
    print(f"   Output: {output_dir}")
    
    # 1. 加载数据
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 3. Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 4. LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # 5. Args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=15, # 快速微调演示
        fp16=True,
        optim="paged_adamw_32bit",
        save_strategy="no",
        report_to="none"
    )
    
    # 6. Formatting
    def formatting_func(example):
        return [f"User: {i}\nAssistant: {o}" for i, o in zip(example['instruction'], example['output'])]
    
    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        max_seq_length=512,
        formatting_func=formatting_func
    )
    
    trainer.train()
    
    # 8. Save
    trainer.model.save_pretrained(output_dir)
    print("✅ Training done!")
    return True