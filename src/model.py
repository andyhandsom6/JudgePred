import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

class LegalPredictionModel(nn.Module):
    """法律判决预测模型（多任务学习）"""
    def __init__(self, model_name="model_zoo/Qwen3-8B", num_charges=321):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 添加被告人特殊token
        self.tokenizer.add_tokens(["<defendant>"])
        self.bert.resize_token_embeddings(len(self.tokenizer))
        
        # PEFT配置 (LoRA)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.bert = get_peft_model(self.bert, peft_config)
        
        # 罪名预测头（多标签分类）
        self.charge_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_charges)
        )
        
        # 刑期预测头（回归）
        self.term_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # 损失函数
        self.charge_loss = nn.BCEWithLogitsLoss()
        self.term_loss = nn.MSELoss()

    def forward(self, input_ids, attention_mask, charge_labels=None, term_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 被告人注意力机制
        defendant_mask = (input_ids == self.tokenizer.convert_tokens_to_ids("<defendant>")).unsqueeze(-1)
        defendant_rep = (sequence_output * defendant_mask).sum(dim=1) / defendant_mask.sum(dim=1).clamp(min=1)
        
        # 罪名预测
        charge_logits = self.charge_head(defendant_rep)
        
        # 刑期预测
        term_preds = self.term_head(defendant_rep)
        
        loss = 0
        if charge_labels is not None:
            loss += self.charge_loss(charge_logits, charge_labels)
            
        if term_labels is not None:
            # 只计算有刑期标签的损失
            valid_mask = (term_labels >= 0).float()
            valid_count = valid_mask.sum().clamp(min=1)
            loss += self.term_loss(term_preds.squeeze() * valid_mask, term_labels) * valid_count
            
        return {
            "charge_logits": charge_logits,
            "term_preds": term_preds.squeeze(),
            "loss": loss
        }
    
    def predict(self, input_text, device="cuda"):
        """预测接口"""
        inputs = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.forward(**inputs)
        
        # 处理罪名预测
        charge_probs = torch.sigmoid(outputs["charge_logits"])
        charge_preds = (charge_probs > 0.4).int().cpu().numpy()
        
        # 处理刑期预测
        term_preds = outputs["term_preds"].clamp(min=0).round().int().cpu().tolist()
        
        return charge_preds, term_preds