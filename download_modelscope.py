from modelscope import snapshot_download
from datasets import load_dataset
from datasets import load_dataset,load_from_disk

#
model_dir = snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct',
                              cache_dir='/Users/baihailong/Documents/modelscope/LLM-Research/qwen2-5_VL_3B_Instruct')