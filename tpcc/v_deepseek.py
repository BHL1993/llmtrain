import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

base_model_name = "/Users/baihailong/PycharmProjects/train/local_model/deepseek_15"


class LogisticsDataset(Dataset):
    def __init__(self, logs, max_seq_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 适配CausalLM
        self.max_seq_len = max_seq_len
        self.processed_data = [self._process_log(log) for log in logs]

    def _time_to_phase(self, hours):
        """阶段划分函数"""
        if hours < 24: return 0
        elif hours < 72: return 1
        else: return 2

    def _process_log(self, log_entries):
        # 时间特征计算
        start_time = pd.to_datetime(log_entries[0]["操作时间"])
        time_deltas, phase_labels, texts = [], [], []
        for entry in log_entries:
            curr_time = pd.to_datetime(entry["操作时间"])
            delta_hours = (curr_time - start_time).total_seconds() / 3600
            time_deltas.append(delta_hours)
            phase_labels.append(self._time_to_phase(delta_hours))
            texts.append(f"{entry['操作类型']}：{entry['内容概述']}")

        # 文本编码
        text_enc = self.tokenizer(
            texts,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 标签生成（假设最后一个步骤是否包含投诉）
        label = 1 if any("投诉" in t for t in texts) else 0

        return {
            "time_deltas": torch.FloatTensor(time_deltas),
            "phase_labels": torch.LongTensor(phase_labels),
            "input_ids": text_enc["input_ids"].squeeze(),
            "attention_mask": text_enc["attention_mask"].squeeze(),
            "labels": torch.FloatTensor([label])
        }

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


import torch.nn as nn


class EnhancedTimeLLM(nn.Module):
    def __init__(self):
        super().__init__()
        # 文本编码器
        self.text_encoder = AutoModelForCausalLM.from_pretrained(base_model_name)
        text_embed_dim = self.text_encoder.config.hidden_size

        # 时间特征编码
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU()
        )
        self.phase_embed = nn.Embedding(3, 128)

        # 特征融合层
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128 + text_embed_dim,
                nhead=8,
                dim_feedforward=1024
            ),
            num_layers=4
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128 + text_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, time_deltas, phase_labels, input_ids, attention_mask):
        # 文本特征
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_features = text_outputs.hidden_states[-1]  # [B, L, D]

        # 时间特征
        time_features = self.time_embed(time_deltas.unsqueeze(-1))  # [B, L, 128]
        phase_features = self.phase_embed(phase_labels)  # [B, L, 128]
        temporal_features = time_features + phase_features

        # 特征拼接
        combined = torch.cat([text_features, temporal_features], dim=-1)
        combined = combined.permute(1, 0, 2)  # [L, B, D]

        # 特征融合
        fused = self.fusion_transformer(combined)

        # 分类预测
        output = self.classifier(fused[-1, :, :])
        return output


def collate_fn(batch):
    """自定义批次处理（处理变长序列）"""
    return {
        "time_deltas": torch.stack([x["time_deltas"] for x in batch]),
        "phase_labels": torch.stack([x["phase_labels"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch])
    }


def train():
    # 示例数据
    sample_logs = [
        [
            {"操作时间": "2024-10-10 10:10:10", "操作类型": "客户进线", "内容概述": "快递破损"},
            {"操作时间": "2024-10-13 11:10:10", "操作类型": "投诉", "内容概述": "未解决"}
        ],
        [
            {"操作时间": "2024-10-11 09:00:00", "操作类型": "客户进线", "内容概述": "包裹未送达"},
            {"操作时间": "2024-10-11 12:00:00", "操作类型": "处理完成", "内容概述": "已重新派送"}
        ]
    ]

    # 数据加载
    dataset = LogisticsDataset(sample_logs)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # 模型初始化
    model = EnhancedTimeLLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))  # 处理类别不平衡

    # 训练循环
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            outputs = model(
                time_deltas=batch["time_deltas"],
                phase_labels=batch["phase_labels"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            loss = criterion(outputs.squeeze(), batch["labels"].squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss / len(dataloader):.4f}")

    # 模型保存
    torch.save(model.state_dict(), "timellm_risk_predictor.pth")

if __name__ == "__main__":
    train()