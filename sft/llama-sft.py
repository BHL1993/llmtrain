import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch

model_name = "/Users/baihailong/Documents/modelscope/LLM-Research/Llama-3.2-1B-Instruct"
# model_name = "/Users/baihailong/PycharmProjects/train/local_model/deepseek_15"


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_hidden, stats_hidden):
        # text_hidden: [batch, seq_len, hidden]
        # stats_hidden: [batch, hidden]

        # 扩展统计特征维度
        stats_hidden = stats_hidden.unsqueeze(1)  # [batch, 1, hidden]

        # 计算注意力
        Q = self.q_proj(text_hidden)
        K = self.k_proj(stats_hidden)
        V = self.v_proj(stats_hidden)

        attn_weights = torch.matmul(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V)  # [batch, seq_len, hidden]
        return self.out_proj(attn_output)


class QwenWithCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size

        # 统计特征编码器
        self.stats_encoder = StatsEncoder(hidden_size=self.hidden_size)

        # 交叉注意力层（查询来自文本，键值来自统计特征）
        self.cross_attn = CrossAttentionLayer(self.hidden_size)

        # 回归预测头
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_size, 2),
            nn.Sigmoid()
        )

    def forward(self, input_ids, stats_features, attention_mask=None, labels=None):
        # 文本编码 [batch, seq_len, hidden]
        text_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_hidden = text_outputs.hidden_states[-1]

        # 统计特征编码 [batch, hidden]
        stats_hidden = self.stats_encoder(stats_features)

        # 交叉注意力融合
        attn_output = self.cross_attn(text_hidden,stats_hidden)

        # 取[CLS]位置预测概率
        pooled = attn_output[:, 0, :]  # 假设第一个token是特殊标记
        return self.reg_head(pooled)


class StatsEncoder(nn.Module):
    """将统计特征编码到与文本隐藏层相同维度"""
    def __init__(self, stats_dim=3, hidden_size=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stats_dim, 256),
            nn.GELU(),
            nn.Linear(256, hidden_size)
        )

    def forward(self, stats):
        return self.mlp(stats)  # [batch, hidden_size]


from peft import LoraConfig, get_peft_model

# 配置LoRA（仅微调Qwen的注意力层）
lora_config = LoraConfig(
    use_dora=False,  # to use Dora OR compare to Lora just set the --use_dora
    r=8,  # Rank of matrix
    lora_alpha=16,
    target_modules=(
        ["k_proj", "v_proj"]
    ),
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["stats_encoder", "cross_attn", "reg_head"],  # 新模块全参数训练
)

model = QwenWithCrossAttn()
model.to(torch.device('cpu'))
model = get_peft_model(model, lora_config)
model.to(torch.device('cpu'))
model.print_trainable_parameters()  # 输出可训练参数占比

from torch.utils.data import Dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


class MultimodalDataset(Dataset):
    def __init__(self, texts, stats, labels, max_length=256):
        self.texts = texts
        self.stats = stats
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "stats_features": torch.FloatTensor(self.stats[idx]),
            "labels": torch.FloatTensor([self.labels[idx]])
        }


# 训练数据示例
train_texts = ["用户多次投诉未解决...", "客服响应迅速..."]
train_stats = [[5, 0.2, 48], [1, 0.05, 12]]  # 历史舆情量、舆情率、处理时效
train_labels = [0.92, 0.15]

# 验证数据
val_texts = ["物流延迟严重...", "服务态度优秀..."]
val_stats = [[3, 0.15, 36], [0, 0.0, 6]]
val_labels = [0.75, 0.05]

from transformers import TrainingArguments, Trainer
import torch


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 分离输入
        stats = inputs.pop("stats_features")
        inputs = {k: v.to(torch.device('cpu')) for k, v in inputs.items()}
        stats = stats.to(torch.device('cpu'))

        # 前向传播
        outputs = model(
            input_ids=inputs["input_ids"],
            stats_features=stats,
            attention_mask=inputs["attention_mask"]
        )

        # 计算MSE损失
        loss = torch.nn.MSELoss()(outputs.squeeze(), inputs["labels"].squeeze())
        return (loss, outputs) if return_outputs else loss


# 训练参数配置
training_args = TrainingArguments(
    output_dir='./stats_result',
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=0,
    push_to_hub=False,
    use_cpu=True,
    #     cpu=True,
    #     hub_model_id=hub_model_id,
    #     gradient_accumulation_steps=16,
    #     fp16=False,#如果用gpu,fp16可以设置为True，低精度浮点数
    learning_rate=3e-4,
)

# 初始化数据集
train_dataset = MultimodalDataset(train_texts, train_stats, train_labels)
val_dataset = MultimodalDataset(val_texts, val_stats, val_labels)

# 开始训练
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()