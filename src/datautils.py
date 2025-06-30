import json

with open("./data/train.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

new_data = []
for i in range(10):
    for j in range(100):
        fact = data[i*10+j]["fact"]
        prompt = f"""
假设你是一个法律专家。请你将以下犯罪事实进行整理。如果字数小于3000字，无需改动，将原文直接输出；如果高于3000字，压缩到3000字左右。要求：
1）保留可以用来断定每个犯罪嫌疑人罪名和量刑的所有关键信息；
2）不要对人名进行任何程度的改动与删减；
3）不要对事实进行任何程度的改动与删减；
4）不需要给出“好的”之类的答复，而是直接输出整理后的事实。
5）所有输出严禁换行，严禁使用Markdown语法
6）所有内容输出在同一行，输出纯文本。
以下是犯罪事实。
{fact}
"""
        new_data.append({
            "custom_id": i*100+j+1,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body":{
                "model": "qwen-max-latest",
                "messages": [
                    {
                        "role":"system",
                        "content":"You are a helpful assistant."
                    },
                    {
                        "role":"user",
                        "content":prompt,
                    }
                ]
            }
        })
    with open(f"./data/qwen_ready/data_patch_{i}.jsonl", "w") as f:
        for item in new_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")