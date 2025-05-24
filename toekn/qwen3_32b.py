from transformers import AutoTokenizer

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")


# 检查目标标签是否已存在于词表
print("<think>" in tokenizer.get_vocab())  # 假设输出True
print("</think>" in tokenizer.get_vocab())  # 假设输出True
print("<reasoning>" in tokenizer.get_vocab())  # 假设输出False
print("</reasoning>" in tokenizer.get_vocab())  # 假设输出False