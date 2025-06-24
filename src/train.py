import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import LegalPredictionModel
import numpy as np
from tqdm import tqdm
import os
import json

# 加载处理后的数据
train_data = load_from_disk("data/train_processed")
test_data = load_from_disk("data/test_processed")

# 加载罪名映射
with open("data/charges.json", "r", encoding="utf-8") as f:
    charge_to_id = json.load(f)
id_to_charge = {v: k for k, v in charge_to_id.items()}

def collate_fn(batch):
    """自定义批处理函数"""
    inputs = [item["input"] for item in batch]
    charges = [item["charges"] for item in batch]
    imprisonments = [item["imprisonments"] for item in batch]
    
    # 标记化
    tokenizer = model.tokenizer
    inputs = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    
    # 处理罪名标签
    charge_labels = torch.zeros(len(batch), len(charge_to_id))
    for i, charge_str in enumerate(charges):
        defendant_charges = charge_str.split(";")
        for j, charges_list in enumerate(defendant_charges):
            for charge in charges_list.split(","):
                if charge in charge_to_id:
                    charge_labels[i, charge_to_id[charge]] = 1
    
    # 处理刑期标签（填充为-1）
    max_defendants = max(len(imp) for imp in imprisonments)
    max_charges_per_def = max(max(len(terms) for terms in imp) for imp in imprisonments)
    
    term_labels = torch.full(
        (len(batch), max_defendants, max_charges_per_def), 
        -1.0, dtype=torch.float
    )
    
    for i, imp in enumerate(imprisonments):
        for j, terms in enumerate(imp):
            for k, term in enumerate(terms):
                term_labels[i, j, k] = term
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "charge_labels": charge_labels,
        "term_labels": term_labels
    }

# 初始化模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LegalPredictionModel().to(device)

# 数据加载器
train_loader = DataLoader(
    train_data, 
    batch_size=4, 
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4
)

val_loader = DataLoader(
    test_data, 
    batch_size=4, 
    collate_fn=collate_fn,
    num_workers=4
)

# 优化器
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100,
    num_training_steps=len(train_loader)*10
)

# 训练循环
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            charge_labels=batch["charge_labels"],
            term_labels=batch["term_labels"]
        )
        
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    charge_preds = []
    charge_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                charge_labels=batch["charge_labels"],
                term_labels=batch["term_labels"]
            )
            
            total_loss += outputs["loss"].item()
            
            # 收集罪名预测
            charge_probs = torch.sigmoid(outputs["charge_logits"])
            charge_preds.extend((charge_probs > 0.4).int().cpu().numpy())
            charge_labels.extend(batch["charge_labels"].cpu().numpy())
    
    # 计算罪名F1
    from sklearn.metrics import f1_score
    f1 = f1_score(charge_labels, charge_preds, average="macro")
    
    return total_loss / len(loader), f1

# 训练主循环
best_f1 = 0
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_f1 = evaluate(model, val_loader, device)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    # 保存最佳模型
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model")
    
    # 学习率调整
    scheduler.step(val_loss)

print("Training completed!")