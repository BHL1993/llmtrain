import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_name = "/Users/baihailong/PycharmProjects/train/local_model/deepseek_15"

# ======================
# 数据预处理组件
# ======================


class LogisticsDataset(Dataset):
    def __init__(self, sequences, max_seq_len=50):
        """
        sequences: [
            {
                "events": [
                    {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                    {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "要求提供照片"}
                ],
                "label": 1
            },
            ...
        ]
        """
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # 预处理操作类型
        all_actions = [e['operation_type'] for seq in sequences for e in seq['events']]
        self.action_encoder = LabelEncoder().fit(all_actions)

        # 构建样本
        self.samples = []
        for seq in sequences:
            processed = self._process_sequence(seq)
            self.samples.append(processed)

    def _process_sequence(self, seq):
        # 时间差计算（小时）
        timestamps = pd.to_datetime([e['operation_time'] for e in seq['events']])
        deltas = [0.0] + [(timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
                          for i in range(1, len(timestamps))]

        # 操作类型编码
        actions = self.action_encoder.transform([e['operation_type'] for e in seq['events']])

        # 文本编码
        text_inputs = self.tokenizer(
            [e['content'] for e in seq['events']],
            padding='max_length',
            truncation=True,
            max_length=32,
            return_tensors='pt'
        )

        # 序列填充
        pad_len = self.max_seq_len - len(seq['events'])
        return {
            'deltas': torch.FloatTensor(deltas + [0] * pad_len),
            'actions': torch.LongTensor(actions.tolist() + [0] * pad_len),
            'text_inputs': {
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask']
            },
            'labels': torch.LongTensor([seq['label']]),
            'mask': torch.BoolTensor([1] * len(seq['events']) + [0] * pad_len)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ======================
# 多模态融合模型
# ======================

class MultiModalClassifier(nn.Module):
    def __init__(self, num_actions, hidden_dim=128):
        super().__init__()
        # 时间特征编码
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.LayerNorm(16)
        )

        # 操作类型编码
        self.action_embed = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=32
        )

        # 文本编码
        self.text_encoder = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.text_proj = nn.Linear(768, 64)

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(16 + 32 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 时序处理层
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=256
            ),
            num_layers=2
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, inputs):
        # 时间特征 [B, L, 16]
        time_feat = self.time_encoder(inputs['deltas'].unsqueeze(-1))

        # 操作类型 [B, L, 32]
        action_feat = self.action_embed(inputs['actions'])

        # 文本特征 [B, L, 64]
        text_outputs = self.text_encoder(
            input_ids=inputs['text_inputs']['input_ids'],
            attention_mask=inputs['text_inputs']['attention_mask'],
            output_hidden_states=True
        )
        text_feat = self.text_proj(text_outputs.last_hidden_state[:, 0, :])

        # 特征拼接 [B, L, 112]
        combined = torch.cat([time_feat, action_feat, text_feat], dim=-1)
        fused = self.fusion(combined)

        # 时序编码
        temporal_out = self.temporal_encoder(
            fused.permute(1, 0, 2),
            src_key_padding_mask=~inputs['mask']
        ).permute(1, 0, 2)

        # 聚合特征
        pooled = temporal_out.mean(dim=1)

        # 分类
        return self.classifier(pooled)


# ======================
# 训练流程
# ======================

def train_model(train_data, val_data, num_epochs=10):
    # 数据加载
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    # 初始化模型
    model = MultiModalClassifier(
        num_actions=len(train_data.action_encoder.classes_))
    model.to(device)

    # 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(num_epochs):
    # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            # batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch['labels'].squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    # 验证阶段
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=-1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch['labels'].cpu().numpy().flatten())

    # 计算指标
    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss / len(train_loader):.4f}")
    print(f"Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

    # 保存最佳模型
    if f1 > best_f1:
        best_f1 = f1
    torch.save(model.state_dict(), "best_model.pth")

    # ======================
    # 使用示例
    # ======================

if __name__ == "__main__":
    # 示例数据
    sample_data = [
            {
                "events": [
                    {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                    {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "请提供照片"}
                ],
                "label": 1
            },
            {
                "events": [
                    {"operation_time": "2024-05-01 10:00", "operation_type": "进线", "content": "咨询进度"},
                    {"operation_time": "2024-05-01 11:00", "operation_type": "回复", "content": "已处理"}
                ],
                "label": 0
            },
            {
                "events": [
                    {"operation_time": "2024-06-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                    {"operation_time": "2024-06-08 11:00", "operation_type": "回复", "content": "请提供照片"}
                ],
                "label": 1
            },
            {
                "events": [
                    {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                    {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "请提供照片"}
                ],
                "label": 1
            },
            {
                "events": [
                    {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                    {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "请提供照片"}
                ],
                "label": 1
            },
            {
                "events": [
                    {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                    {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "请提供照片"}
                ],
                "label": 1
            },
            {
                "events": [
                    {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                    {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "请提供照片"}
                ],
                "label": 1
            },
            # 更多样本...
    ]

    # 数据集划分
    train_data, val_data = train_test_split(sample_data, test_size=0.2)

    # 创建数据集
    train_dataset = LogisticsDataset(train_data)
    val_dataset = LogisticsDataset(val_data)

    # 开始训练
    train_model(train_dataset, val_dataset)