# from datasets import load_dataset,load_from_disk
# #下载数据集
# data_path='timdettmers/openassistant-guanaco'
# dataset = load_dataset(data_path)
#
# # 保存数据集
# dataset.save_to_disk('./openassistant-guanaco/')



from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaConfig

# 选择你想要下载的模型名称，例如 "gpt2"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "Qwen/Qwen2.5-1.5B"
model_name = "huggyllama/llama-7b"
#
# # 下载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# # 保存到本地
# model.save_pretrained("./local_model/llama_7b")
# tokenizer.save_pretrained("./local_model/llama_7b")


llama_config = LlamaConfig.from_pretrained('/Users/baihailong/PycharmProjects/train/local_model/llama-7b')
llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    '/Users/baihailong/PycharmProjects/train/local_model/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=llama_config,
                )
print('')