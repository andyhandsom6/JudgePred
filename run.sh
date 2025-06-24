#!/bin/bash

# 数据预处理
echo "Preprocessing data..."
python src/preprocess.py

# 训练模型
echo "Training model..."
torchrun --nproc_per_node=8 src/train.py

# 生成预测
echo "Generating predictions..."
python src/inference.py

# 评估结果
echo "Evaluating..."
python -c "
from metric_charge import score
import pandas as pd

# 加载真实标签
solution = pd.read_csv('data/sample_submission_accusation.csv')
submission = pd.read_csv('submission.csv')

# 计算分数
f1 = score(solution, submission, 'id')
print(f'Case-level F1 Score: {f1:.4f}')
"