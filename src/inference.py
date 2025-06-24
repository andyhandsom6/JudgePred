import torch
import json
from model import LegalPredictionModel
from transformers import AutoTokenizer
from preprocess import create_prompt, load_articles
from tqdm import tqdm
import numpy as np

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LegalPredictionModel()
model.load_state_dict(torch.load("best_model.pth"))
model = model.to(device)
model.eval()

# 加载罪名映射
with open("data/charges.json", "r", encoding="utf-8") as f:
    charge_to_id = json.load(f)
id_to_charge = {v: k for k, v in charge_to_id.items()}

# 加载测试数据
with open("data/test.jsonl", "r", encoding="utf-8") as f:
    test_data = [json.loads(line) for line in f]

# 加载法条
articles = load_articles("data/articles.json")

# 生成预测结果
results = []
for item in tqdm(test_data, desc="Predicting"):
    # 创建输入
    input_text = create_prompt(item, articles)
    
    # 获取预测
    charge_preds, term_preds = model.predict(input_text, device)
    
    # 转换罪名ID为名称
    charge_names = []
    for pred in charge_preds:
        charges = []
        for i, val in enumerate(pred):
            if val == 1:
                charges.append(id_to_charge[i])
        charge_names.append(",".join(charges))
    
    # 格式化刑期预测
    formatted_terms = []
    for terms in term_preds:
        # 过滤无效预测
        valid_terms = [int(t) for t in terms if t >= 0]
        formatted_terms.append(valid_terms)
    
    results.append({
        "id": item["id"],
        "charges": ";".join(charge_names),
        "imprisonments": formatted_terms
    })

# 保存结果
with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Prediction saved to submission.json")