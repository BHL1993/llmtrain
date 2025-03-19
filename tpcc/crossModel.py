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

# 时间特征编码维度，约为文本特征的1/6，防止时序信息被淹没
time_embed_dim = 128


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


class EnhancedLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载基础llm模型
        self.base_llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        text_embed_dim = self.base_llm_model.config.hidden_size

        # 时间特征编码
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU()
        )

        # 阶段感知位置编码
        self.phase_embed = nn.Embedding(3, time_embed_dim)  # 3个阶段

        # 文本编码与（时间特征编码+阶段感知位置编码）自注意力融合Transformer
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=time_embed_dim + text_embed_dim,
                nhead=8,
                dim_feedforward=1024
            ),
            num_layers=4
        )

    def forward(self, time_deltas, phase_labels, input_ids, attention_mask):
        # 文本特征
        text_features = self.base_llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).last_hidden_state  # [B, L, D]

        # 时间特征
        time_features = self.time_embed(time_deltas.unsqueeze(-1))  # [B, L, 128]
        phase_features = self.phase_embed(phase_labels)  # [B, L, 128]
        temporal_features = time_features + phase_features

        # 特征拼接
        combined = torch.cat([text_features, temporal_features], dim=-1)

        # 增强的位置编码
        combined = combined.permute(1, 0, 2)  # [L, B, D]
        fused = self.fusion_transformer(combined)

        return fused


class YuQingEnhanceLLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载增强文本模型
        self.txt_model = EnhancedLLM()
        # 增强文本模型隐状态维度
        self.hidden_size = self.txt_model.base_llm_model.config.hidden_size + time_embed_dim

        # 统计特征编码器
        self.stats_encoder = StatsEncoder(hidden_size=self.hidden_size)

        # 交叉注意力层（查询来自文本，键值来自统计特征）
        self.cross_attn = CrossAttentionLayer(self.hidden_size)

        # 回归预测头
        self.reg_head = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, stats_features, attention_mask=None, time_deltas=None, phase_labels=None, labels=None):
        # 文本编码
        text_hidden_states = self.txt_model(
            time_deltas=time_deltas,
            phase_labels=phase_labels,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 统计特征编码 [batch, hidden]
        stats_hidden = self.stats_encoder(stats_features)

        # 交叉注意力融合
        attn_output = self.cross_attn(text_hidden_states, stats_hidden)

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
    modules_to_save=["stats_encoder", "cross_attn", "reg_head", "time_embed", "phase_embed", "fusion_transformer"], # 新模块全参数训练
)

model = YuQingEnhanceLLMModel()
model.to(torch.device('cpu'))
model = get_peft_model(model, lora_config)
model.to(torch.device('cpu'))
model.print_trainable_parameters()  # 输出可训练参数占比

tokenizer = AutoTokenizer.from_pretrained(model_name)


class MultimodalDataset(Dataset):
    def __init__(self, texts, stats, labels, max_length=4096):
        self.texts = texts
        self.stats = stats
        self.labels = labels
        self.max_length = max_length

    def _time_to_phase(self, hours):
        """阶段划分函数"""
        if hours < 24:
            return 0
        elif hours < 72:
            return 1
        else:
            return 2

    def _process_log(self, log_entries):
        # 时间特征计算
        start_time = pd.to_datetime(log_entries[0]["操作时间"])
        time_deltas = []
        phase_labels = []
        texts = []

        for entry in log_entries:
            curr_time = pd.to_datetime(entry["操作时间"])
            delta_hours = (curr_time - start_time).total_seconds() / 3600
            time_deltas.append(delta_hours)
            phase_labels.append(self._time_to_phase(delta_hours))
            texts.append(f"{entry['操作类型']}：{entry['内容概述']}")

        return texts, time_deltas, phase_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        texts, time_deltas, phase_labels = self._process_log(self.texts[idx])
        encoding = tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "stats_features": torch.FloatTensor(self.stats[idx]),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "time_deltas": torch.FloatTensor(time_deltas),
            "phase_labels": torch.LongTensor(phase_labels),
            "labels": torch.FloatTensor([self.labels[idx]])
        }


# 训练数据示例
train_texts = [
    [
        {
            "操作时间": "2024-10-10 10:10:10",
            "操作类型": "客户进线",
            "内容概述": "快递破损"
        },
        {
            "操作时间": "2024-10-13 11:10:10",
            "操作类型": "投诉",
            "内容概述": "未解决"
        }
    ],
    [
        {
            "操作时间": "2024-11-10 12:10:10",
            "操作类型": "客户进线",
            "内容概述": "快递丢失"
        },
        {
            "操作时间": "2024-12-15 11:10:10",
            "操作类型": "投诉",
            "内容概述": "已解决"
        }
    ]
]
train_stats = [[5, 0.2, 48], [1, 0.05, 12]]  # 历史舆情量、舆情率、处理时效
train_labels = [1, 0]

# 验证数据
val_texts = [
    [
        {
            "操作时间": "2024-10-10 10:10:10",
            "操作类型": "客户进线",
            "内容概述": "快递破损"
        },
        {
            "操作时间": "2024-10-13 11:10:10",
            "操作类型": "投诉",
            "内容概述": "未解决"
        }
    ],
    [
        {
            "操作时间": "2024-10-10 10:10:10",
            "操作类型": "客户进线",
            "内容概述": "快递破损"
        },
        {
            "操作时间": "2024-10-13 11:10:10",
            "操作类型": "投诉",
            "内容概述": "未解决"
        }
    ]
]
val_stats = [[3, 0.15, 36], [0, 0.0, 6]]
val_labels = [1, 0]


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
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=0,
    push_to_hub=False,
    no_cuda=True,
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