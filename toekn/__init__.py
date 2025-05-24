import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
import pandas as pd

# model_name = "/Users/baihailong/Documents/modelscope/LLM-Research/Llama-3.2-1B-Instruct"
model_name = "/Users/baihailong/PycharmProjects/train/local_model/deepseek_15"

tokenizer = AutoTokenizer.from_pretrained(model_name)


encoding = tokenizer(
            "以下为思维链内容，请你按照思维链的逻辑顺序进行判断。注意",
            max_length=2000,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

print(encoding)