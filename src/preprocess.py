import json
import re
from collections import defaultdict
import pandas as pd
from datasets import Dataset

def extract_entities(text):
    """使用规则抽取法律实体（传统NLP技术增强）"""
    entities = {
        "defendants": [],
        "locations": [],
        "times": [],
        "methods": [],
        "consequences": []
    }
    
    # 被告人抽取（匹配中文姓名模式）
    name_matches = re.findall(r'被告人[：: ]?([\u4e00-\u9fa5]{2,4})', text)
    entities["defendants"] = list(set(name_matches))
    
    # 地点抽取
    loc_matches = re.findall(r'在([\u4e00-\u9fa5]{2,10}?(?:市|县|区|镇|村|路|街道))', text)
    entities["locations"] = list(set(loc_matches))
    
    # 时间抽取
    time_matches = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日)', text)
    entities["times"] = list(set(time_matches))
    
    return entities

def load_articles(articles_path):
    """加载刑法条文知识库"""
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles

def create_prompt(example, articles):
    """创建增强的输入提示"""
    fact = example["fact"]
    defendants = example["defendants"]
    
    # 实体抽取
    entities = extract_entities(fact)
    
    # 匹配相关法条
    related_articles = []
    for art_id, content in articles.items():
        if any(keyword in fact for keyword in ["抢劫", "盗窃", "诈骗"]):  # 简化的关键词匹配
            related_articles.append(content[:100] + "...")
    
    # 构建结构化提示
    prompt = f"""【案件事实】
{fact}

【结构化信息】
被告人: {", ".join(defendants)}
地点: {", ".join(entities["locations"])}
时间: {", ".join(entities["times"])}
相关法条: 
{chr(10).join(related_articles[:3])}

【任务要求】
1. 预测每个被告人的罪名（多个罪名用逗号分隔）
2. 预测每个被告人每个罪名的刑期（月）
"""
    return prompt

def parse_label(example):
    """解析标签数据"""
    outcomes = example["outcomes"]
    charges = []
    imprisonments = []
    
    for defendant in outcomes:
        name = defendant["name"]
        judgments = defendant["judgment"]
        def_charges = []
        def_terms = []
        
        for judgment in judgments:
            if judgment["standard_accusation"]:
                def_charges.append(judgment["standard_accusation"])
                def_terms.append(int(judgment["imprisonment"]) if judgment["imprisonment"] else 0)
        
        charges.append(",".join(def_charges))
        imprisonments.append(def_terms)
    
    return {
        "charges": ";".join(charges),
        "imprisonments": imprisonments
    }

def preprocess_data(data_path, articles_path):
    """预处理数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    articles = load_articles(articles_path)
    processed = []
    
    for example in data:
        prompt = create_prompt(example, articles)
        label = parse_label(example)
        
        processed.append({
            "id": example.get("id", ""),
            "input": prompt,
            "charges": label["charges"],
            "imprisonments": label["imprisonments"]
        })
    
    return Dataset.from_pandas(pd.DataFrame(processed))

if __name__ == "__main__":
    train_data = preprocess_data("data/train.jsonl", "data/articles.json")
    test_data = preprocess_data("data/test.jsonl", "data/articles.json")
    
    train_data.save_to_disk("data/train_processed")
    test_data.save_to_disk("data/test_processed")