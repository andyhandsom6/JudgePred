import json
import csv
import argparse

parser = argparse.ArgumentParser(description="Convert JSON results to CSV")
parser.add_argument("--task", type=int, default=1, help="Task identifier (default: 1)")
args = parser.parse_args()

model_name = "qwen_0.6B_full_bs_1_grac_8_lr_2e-5_epoch_5_max_5120_t2_short/epoch_3"

with open(f"./results/{model_name}/results.json", "r") as f:
    data = json.load(f)


# 输出 CSV 文件名
output_file = f"./results/{model_name}/results.csv"

with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if args.task == 1:
        writer.writerow(['id', 'accusations'])
    else:
        writer.writerow(['id', 'imprisonment'])

    for idx, case_str in enumerate(data, start=1):
        case_list = case_str  # 把字符串转为Python对象
        record = []

        for person in case_list:
            # 每个人的 judgment 里可能有多个 accusation
            if args.task == 1:
                person_accusations = [j['standard_accusation'] for j in person['judgment']]
                record.append(','.join(person_accusations))  # 同一个人多罪名用逗号隔开
            else:
                try:
                    person_imprisonment = [j['imprisonment'] for j in person['judgment']]
                except:
                    print(case_list)
                # record.append(','.join(person_imprisonment))  # 同一个人多罪名用逗号隔开
                record.append(person_imprisonment)

        # 所有人的罪名用分号隔开，并加引号
        if args.task == 1: 
            record = ';'.join(record)
        else:
            record = str(record)
        writer.writerow([idx, f'{record}'])

print(f"转换完成，结果保存为 {output_file}")
