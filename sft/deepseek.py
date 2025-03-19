import os

import torch
# from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModel
)
from datasets import load_dataset,load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
#
base_model='/Users/baihailong/Documents/modelscope/LLM-Research/Llama-3.2-1B-Instruct/'

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
def train(model,dataset,data_collator):
    # get the peft model with LoRa config
    model = get_peft_model(model, lora_config)
    model.to(torch.device('cpu'))#如果用cuda替换成cuda:0


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
if __name__=='__main__':

    # tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    tokenizer = AutoTokenizer.from_pretrained("./local_model/deepseek_15")
    model = AutoModelForCausalLM.from_pretrained("./local_model/deepseek_15")


    #从本地加载
    dataset=load_from_disk('./openassistant-guanaco/')
    #重要，原本的词表中没有[PAD]token
    tokenizer.pad_token = tokenizer.eos_token
    instruction_template = "### Human:"
    response_template = "### Assistant:"
    data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer,mlm=False)
    train(model,dataset,data_collator)
    # 继续预训练
    # def tokenize_function(examples):
    #         inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    #         inputs["labels"] = inputs["input_ids"].copy()  # setting labels for a language modeling task
    #         return inputs

    # Tokenize the dataset and prepare for training
    # tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    # Data collator to dynamically pad the batched examples
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # train(model,tokenized_datasets,data_collator)