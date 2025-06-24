from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/raid/data/anqi"  # 替换为你希望保存模型的本地路径

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", cache_dir=cache_dir)
