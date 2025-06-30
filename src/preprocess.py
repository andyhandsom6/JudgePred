import json
# import re
# from collections import defaultdict
import pandas as pd
from datasets import Dataset
import argparse

def load_charges(charges_path):
    """加载刑法条文知识库"""
    with open(charges_path, 'r', encoding='utf-8') as f:
        charges = json.load(f)
    return charges

def create_prompt(example, charges, args):
    """创建增强的输入提示"""
    fact = example["fact"]
    defendants = example["defendants"]
    query = (
        "请根据上述案件事实，直接以json的形式给出每名被告人的罪名。"
        if args.task == 1 else
        "请根据上述案件事实，直接以json的形式给出每名被告人的罪名和刑期（单位：月）。"
    )

    prompt = f"""【案件事实】
{fact}
【被告人信息】
{defendants}
{query}
"""
    return prompt

def parse_label(example, args):
    """解析标签数据 - 修复刑期标签处理"""
    outcomes = example["outcomes"]

    for people in outcomes:
        for judgment in people["judgment"]:
            if args.task == 1:
                del judgment["standard_accusation"]
    
    return outcomes

def preprocess_data(data_path, charges_path, args):
    """预处理数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    charges = load_charges(charges_path)
    processed = []
    
    for example in data:
        try:
            prompt = create_prompt(example, charges, args)
            label = parse_label(example, args)
            
            processed.append({
                "prompt": prompt,
                "label": str(label),
            })
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    return processed
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for JudgePred")
    parser.add_argument("--task", type=int, default=1)
    args = parser.parse_args()

    train_data = preprocess_data("data/train.jsonl", "data/charges.json", args)
    test_data = preprocess_data("data/test.jsonl", "data/charges.json", args)
    
    with open(f"data/train_processed_q{args.task}.json", "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(f"data/test_processed_q{args.task}.json", "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)