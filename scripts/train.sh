torchrun --nproc_per_node=8 src/train.py \
    --task 2 \
    --save_path ./checkpoints/qwen_0.6B_full_bs_1_grac_8_lr_2e-5_epoch_7_max_5120_t2_add_article_month_cloze \
    --add_article True \