from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

def load_lora_model(model_path="model_zoo/Qwen3-4B", lora_rank=16):
    """
    加载基础模型并添加LoRA适配器
    Args:
        model_path: Qwen3模型路径或标识符
        lora_rank: LoRA秩大小
    Returns:
        model: 配置了LoRA的模型
    """
    print("base model path:", model_path)
    # 加载基础模型 (使用bfloat16加速训练并减少显存)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True,
    )
    
    # # 配置LoRA参数
    # lora_config = LoraConfig(
    #     r=lora_rank,
    #     lora_alpha=32,
    #     target_modules=["q_proj", "v_proj"],# ["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_dropout=0.05,
    #     task_type="CAUSAL_LM",
    #     inference_mode=False
    # )
    
    # # 添加LoRA适配器
    # model = get_peft_model(model, lora_config)

    # for name, param in model.named_parameters():
    #     if (
    #         "layers.30." in name or "layers.31." in name or
    #         "layers.32." in name or "layers.33." in name or 
    #         "layers.34." in name or "layers.35." in name
    #     ):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    # model.print_trainable_parameters()  # 打印可训练参数占比
    return model