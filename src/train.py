import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from accelerate import Accelerator
from tqdm import tqdm
from model import load_lora_model
import os
import argparse
import random

# 配置参数
MODEL_PATH = "model_zoo/Qwen3-0.6B" # "model_zoo/Qwen3-8B"
# DATA_PATH = "data/train_processed_q1.json"
LORA_RANK = 16
BATCH_SIZE = 1  # 每GPU批大小
GRAD_ACCUM_STEPS = 8  # 梯度累积步数
LEARNING_RATE = 2e-5
EPOCHS = 5
MAX_LENGTH = 5120  # 最大长度

class FineTuneDataset(Dataset):
    """微调数据集类（使用Qwen3对话模板）"""
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 定义对话模板部分（固定）
        self.system_str = ""
        self.user_start = "<|im_start|>user\n"
        self.user_end = "<|im_end|>\n"
        self.assistant_start = "<|im_start|>assistant\n<think>\n\n</think>\n"
        self.assistant_end = "<|im_end|>"

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        label = item['label']

        ablation = False
        if ablation:
            l = eval(label)
            for people in l:
                random.shuffle(people['judgment'])
            label = str(l)
            # print(label)
        
        # 构造各部分的字符串
        user_str = self.user_start + prompt + self.user_end
        assistant_str = self.assistant_start + label + self.assistant_end
        
        user_ids = self.tokenizer(
            user_str,
            add_special_tokens=False,
            return_tensors=None
        )["input_ids"]
        
        assistant_ids = self.tokenizer(
            assistant_str,
            add_special_tokens=False,
            return_tensors=None
        )["input_ids"]
        
        # 拼接
        input_ids = user_ids + assistant_ids
        # 创建labels，user设置为-100，assistant部分保留
        labels = [-100] * len(user_ids) + assistant_ids
        
        # 如果总长度超过最大长度，则从左侧截断（保留最后max_length个token）
        if len(input_ids) > self.max_length:
            print(f"Exceeding max length: {len(input_ids)} tokens, truncated")
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
        
        # 创建attention_mask（全1）
        attention_mask = [1] * len(input_ids)
        
        # 确保长度一致
        assert len(input_ids) == len(labels) == len(attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 model")
    parser.add_argument("--task", type=int, default=1, help="Task identifier (default: 1)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the fine-tuned model")
    args = parser.parse_args()
    
    DATA_PATH = f"data/train_processed_q{args.task}.json"
    SAVE_DIR = args.save_path  # 保存路径
    os.makedirs(SAVE_DIR, exist_ok=True)  # 确保保存目录存在

    # 初始化加速器 (自动处理分布式训练)
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        mixed_precision="bf16"
    )
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True
    )
    # 注意：Qwen3的pad_token是<|endoftext|>，但通常不需要设置，因为对话模板中已经包含了结束符
    # 不过，为了在padding时使用，我们设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = load_lora_model(MODEL_PATH, LORA_RANK)
    
    # 准备数据集
    dataset = FineTuneDataset(DATA_PATH, tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=True
    )
    
    # 配置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # 使用加速器准备对象
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    
    # 训练循环
    model.train()
    total_steps = len(train_loader) * EPOCHS
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    
    for epoch in range(EPOCHS):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # 梯度裁剪
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
            
            # 每一步或每n步输出loss
            # if accelerator.is_main_process and step % 10 == 0:  # 每10步输出一次
            #     print(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}")
    
        # 保存模型
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        epoch_save_dir = os.path.join(SAVE_DIR, f"epoch_{epoch+1}")
        unwrapped_model.save_pretrained(epoch_save_dir, safe_serialization=True)
        tokenizer.save_pretrained(epoch_save_dir)
    
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} 结束，模型已保存至 {epoch_save_dir}")
        
if __name__ == "__main__":
    main()