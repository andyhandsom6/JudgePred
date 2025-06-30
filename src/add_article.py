import json

with open("./data/articles.json", "r") as f:
    data = json.load(f)

with open("./data/train_processed_q2.json", "r") as f:
    train_data = json.load(f)

article_data = []
for item in data.values():
    title, content = item.split('条：')
    title = title + '条'
    article_data.append({
        "prompt": f"请默写{title}",
        "label": content
    })

for _ in range(5):
    train_data.extend(article_data)

with open("./data/train_processed_q2_article_5x.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)