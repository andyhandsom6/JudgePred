import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.multiprocessing import Process, set_start_method
import math

def run(rank, world_size, data, model_dir, output_path):
    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": rank}
    )
    model.eval()

    result = {}

    for i in tqdm(range(rank, len(data), world_size), desc=f"GPU {rank}"):
        item = data[i]
        prompt = item["prompt"]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        while True:
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
            try:
                result[i] = eval(content)  # 注意加上索引，方便后续还原顺序
                break
            except:
                print(f"Error processing item {i}, retrying...")


    with open(f"{output_path}/part_{rank}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def multi_gpu_run():
    set_start_method("spawn", force=True)
    world_size = torch.cuda.device_count()
    print(f"Launching with {world_size} GPUs")

    with open("./data/test_processed_q2_short.json") as f:
        test_data = json.load(f)

    model_name = "qwen_0.6B_full_bs_1_grac_8_lr_2e-5_epoch_5_max_5120_t2_short/epoch_3"
    model_dir = f"./checkpoints/{model_name}"
    output_path = f"./results"
    os.makedirs(output_path, exist_ok=True)

    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, world_size, test_data, model_dir, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 汇总所有结果并按索引排序
    final_result = [None] * len(test_data)
    for rank in range(world_size):
        part_path = f"{output_path}/part_{rank}.json"
        with open(part_path) as f:
            part_result = json.load(f)
            for k, v in part_result.items():
                final_result[int(k)] = v
        # os.remove(part_path)  # 删除临时文件
    
    os.makedirs(f"{output_path}/{model_name}", exist_ok=True)
    with open(f"{output_path}/{model_name}/results.json", "w") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    multi_gpu_run()
