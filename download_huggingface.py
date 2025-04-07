# from datasets import load_dataset,load_from_disk
# #下载数据集
# data_path='timdettmers/openassistant-guanaco'
# dataset = load_dataset(data_path)
#
# # 保存数据集
# dataset.save_to_disk('./openassistant-guanaco/')



from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaConfig
from huggingface_hub import snapshot_download


# 选择你想要下载的模型名称，例如 "gpt2"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "Qwen/Qwen2.5-1.5B"
# model_name = "huggyllama/llama-7b"
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
#
# # 加载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# # 保存到本地
# model.save_pretrained("./local_model/llama_7b")
# tokenizer.save_pretrained("./local_model/llama_7b")


# 仅下载模型文件
local_dir = snapshot_download(repo_id=model_name, cache_dir="/Users/baihailong/Documents/modelscope/LLM-Research/qwen2-5_VL_3B_Instruct")