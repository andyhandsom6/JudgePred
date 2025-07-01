import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.multiprocessing import Process, set_start_method
import math
import random

def run(rank, world_size, data, model_dir, output_path, long_data_index):
    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        # torch_dtype=torch.bfloat16,
        device_map={"": rank}
    )
    model.eval()

    result = {}

    for i in tqdm(range(rank, len(long_data_index), world_size), desc=f"GPU {rank}"):
        item = data[long_data_index[i]]
        fact = item["fact"]
        prompt = f"""
假设你是一个法律专家。请你将以下犯罪事实压缩到**2000**字左右。要求：
1）保留可以用来断定每个犯罪嫌疑人**罪名**和**量刑**的关键信息；
2）不要对**嫌疑犯**的人名进行任何程度的改动与删减；
3）**可以**删去与判定罪行无关的证据收集、调查过程等内容与人物；
4）不需要给出“好的”之类的答复，而是**直接输出**整理后的事实。
5）所有输出严禁换行，严禁使用Markdown语法
6）所有内容输出在同一行，输出纯文本。
以下是犯罪事实。
{fact}
"""
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        if inputs["input_ids"].shape[1] > 4608:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=32768
                )
            output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            data[long_data_index[i]]["fact"] = content
        result[long_data_index[i]] = data[long_data_index[i]]  # 注意加上索引，方便后续还原顺序

    os.makedirs(f"{output_path}/tmp", exist_ok=True)
    with open(f"{output_path}/tmp/part_{rank}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def multi_gpu_run():
    set_start_method("spawn", force=True)
    world_size = torch.cuda.device_count()
    print(f"Launching with {world_size} GPUs")

    dataset = "test"
    with open(f"./data/{dataset}.jsonl") as f:
        test_data = [json.loads(line) for line in f]

    with open("./data/test_data_tmp_idx.json") as f:
        long_data_index = json.load(f)

    random.shuffle(long_data_index)
    # test_data = test_data[:24]

    model_dir = "model_zoo/Qwen3-0.6B"
    output_path = f"./data/qwen_ready_reorganize_{dataset}"
    os.makedirs(output_path, exist_ok=True)

    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, world_size, test_data, model_dir, output_path, long_data_index))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 汇总所有结果并按索引排序
    final_result = [None] * len(test_data)
    for rank in range(world_size):
        part_path = f"{output_path}/tmp/part_{rank}.json"
        with open(part_path) as f:
            part_result = json.load(f)
            for k, v in part_result.items():
                final_result[int(k)] = v
        # os.remove(part_path)  # 删除临时文件
    
    os.makedirs(f"{output_path}", exist_ok=True)
    with open(f"{output_path}/results.json", "w") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    multi_gpu_run()
