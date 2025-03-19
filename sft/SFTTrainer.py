
from datasets import load_dataset,load_from_disk
###加载模型
# Load model directly
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModel
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM

epochs=3

# # 下载数据集
# data_path='timdettmers/openassistant-guanaco'
# dataset = load_dataset(data_path)
# # 保存数据集
# dataset.save_to_disk('./openassistant-guanaco/')

# 从本地加载
dataset = load_from_disk('./openassistant-guanaco/')

tokenizer = AutoTokenizer.from_pretrained("./local_model/deepseek_15")
model = AutoModelForCausalLM.from_pretrained("./local_model/deepseek_15")

# Lora配置
lora_config = LoraConfig(
    use_dora=False,  # to use Dora OR compare to Lora just set the --use_dora
    r=8,  # Rank of matrix
    lora_alpha=16,
    target_modules=(
        ["k_proj", "v_proj"]
    ),
    lora_dropout=0.05,
    bias="none",
)

# get the peft model with LoRa config
model = get_peft_model(model, lora_config)
model.to(torch.device('cpu'))  # 如果用cuda替换成cuda:0

# Define training arguments
training_args = TrainingArguments(
    output_dir='./llama_lora_ft',
    num_train_epochs=1,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    push_to_hub=False,
    no_cuda=True,
    use_cpu=True,
    #     cpu=True,
    #     hub_model_id=hub_model_id,
    #     gradient_accumulation_steps=16,
    #     fp16=False,#如果用gpu,fp16可以设置为True，低精度浮点数
    learning_rate=3e-4,
)

# 重要，原本的词表中没有[PAD]token
tokenizer.pad_token = tokenizer.eos_token
instruction_template = "### Human:"
response_template = "### Assistant:"
data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                                response_template=response_template, tokenizer=tokenizer, mlm=False)

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)
# Start model training
trainer.train()