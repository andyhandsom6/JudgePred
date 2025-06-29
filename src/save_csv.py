import json
import csv

model_name = "qwen_0.6B_full_bs_1_grac_8_lr_5e-5_epoch_3_max_8192"

with open(f"./results/{model_name}/results.json", "r") as f:
    data = json.load(f)


# 输出 CSV 文件名
output_file = f"./results/{model_name}/results.csv"

with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'accusations'])

    for idx, case_str in enumerate(data, start=1):
        case_list = case_str  # 把字符串转为Python对象
        accusations = []

        for person in case_list:
            # 每个人的 judgment 里可能有多个 accusation
            person_accusations = [j['standard_accusation'] for j in person['judgment']]
            accusations.append(','.join(person_accusations))  # 同一个人多罪名用逗号隔开

        # 所有人的罪名用分号隔开，并加引号
        all_accusations = ';'.join(accusations)
        writer.writerow([idx, f'{all_accusations}'])

print(f"转换完成，结果保存为 {output_file}")
