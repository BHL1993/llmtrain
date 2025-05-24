# -*- coding: utf-8 -*-
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig

# 1. 加载预训练模型和分词器
model_name = "meta-llama/Llama-2-7b-chat-hf"  # 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充符

# 主模型和参考模型（需相同架构）
model = AutoModelForCausalLM.from_pretrained(model_name)
model_ref = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 定义偏好数据集（示例）
# 每条数据包含：输入提示（prompt）、选择的响应（chosen）、拒绝的响应（rejected）
preference_data = [
    {
        "prompt": "Explain quantum computing in simple terms:",
        "chosen": "Quantum computing uses qubits to perform calculations exponentially faster than classical computers.",
        "rejected": "Quantum computing is just a sci-fi concept."
    },
    {
        "prompt": "Write a poem about artificial intelligence:",
        "chosen": "Silent circuits hum, algorithms dance unseen...",
        "rejected": "AI is dangerous and will destroy humanity."
    }
]

# 转换为 Hugging Face Dataset 格式
dataset = Dataset.from_dict({
    "prompt": [d["prompt"] for d in preference_data],
    "chosen": [d["chosen"] for d in preference_data],
    "rejected": [d["rejected"] for d in preference_data],
})

# 3. 配置 DPO 参数
dpo_config = DPOConfig(
    beta=0.1,                   # 控制偏好对齐强度的超参数（默认 0.1-0.5）
    learning_rate=5e-6,         # 学习率
    per_device_train_batch_size=2,  # 每个设备的批次大小
    gradient_accumulation_steps=1,  # 梯度累积步数
    logging_steps=1,            # 日志记录间隔
    output_dir="dpo_finetuned", # 输出目录
    remove_unused_columns=False,  # 禁用自动过滤未使用的列
)

# 4. 初始化 DPO 训练器
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=model_ref,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 5. 执行 DPO 训练
dpo_trainer.train()

# 6. 保存微调后的模型
dpo_trainer.save_model("dpo_finetuned_llama")
tokenizer.save_pretrained("dpo_finetuned_llama")