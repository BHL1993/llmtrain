# 拼接 + 自注意力方案

from datetime import datetime
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

base_model_name = "/Users/baihailong/PycharmProjects/train/local_model/deepseek_15"


def preprocess_log(log_entry, tokenizer):
    events = log_entry['events']
    label = torch.tensor(log_entry['label'], dtype=torch.float)

    # 将operation_time转换为datetime对象，并计算相邻事件之间的时间差
    time_deltas = []
    prev_time = None
    for event in events:
        current_time = datetime.strptime(event['operation_time'], "%Y-%m-%d %H:%M")
        if prev_time is not None:
            delta = (current_time - prev_time).total_seconds() / 3600  # 转换为小时
            time_deltas.append(delta)
        prev_time = current_time

    # 添加第一个事件前的一个虚拟时间差值，假设为0
    time_deltas.insert(0, 0.0)

    # 编码操作类型
    operation_types = {'进线': 0, '回复': 1}  # 根据实际情况扩展
    phase_labels = [operation_types[event['operation_type']] for event in events]

    strList = [event['content'] for event in events]
    str = "".join(strList)
    encodings = tokenizer(str, padding='max_length', truncation=True,
                          max_length=1024, return_tensors='pt')

    return {
        'time_deltas': torch.tensor(time_deltas, dtype=torch.float),
        'input_ids': encodings['input_ids'].squeeze(),
        'attention_mask': encodings['attention_mask'].squeeze(),
        'phase_labels': torch.tensor(phase_labels, dtype=torch.long),
        'label': label
    }


class LogisticsDataset(Dataset):
    def __init__(self, logs):
        self.logs = logs
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        return preprocess_log(self.logs[idx], self.tokenizer)


def collate_fn(batch):
    time_deltas = [item['time_deltas'] for item in batch]
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    phase_labels = [item['phase_labels'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    padded_time_deltas = pad_sequence(time_deltas, batch_first=True)
    time_mask = padded_time_deltas != 0

    padded_phase_labels = pad_sequence(phase_labels, batch_first=True, padding_value=0)

    return {
        'time_deltas': padded_time_deltas,
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'phase_labels': padded_phase_labels,
        'labels': labels,
        'time_mask': time_mask
    }


import torch.nn as nn
from transformers import BertModel, AutoTokenizer, AutoModelForCausalLM


class TimeTextTransformer(nn.Module):
    def __init__(self,time_feature_size=128, num_heads=8, num_layers=6, num_phases=3):
        super(TimeTextTransformer, self).__init__()
        self.text_encoder = AutoModelForCausalLM.from_pretrained(base_model_name)
        # 时间间隔特征、阶段特征为了和文本特征做自注意力，需要对齐维度
        text_feature_size = self.text_encoder.config.hidden_size
        # 文本特征与统计特征维度对齐
        time_feature_size = text_feature_size
        self.time_embed = nn.Linear(1, time_feature_size)
        self.phase_embed = nn.Embedding(num_phases, time_feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=time_feature_size + text_feature_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(time_feature_size + text_feature_size, 1)

    def forward(self, time_deltas, input_ids, attention_mask, phase_labels, time_mask):
        # 文本特征
        # text_encoder_out : [batch_size, seq_len, hidden_dim]
        text_encoder_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # text_features : [batch_size, seq_len, hidden_dim]
        text_features = text_encoder_out.hidden_states[-1]

        # 时间特征
        # time_deltas : [batch_size, event_num]
        # time_deltas.unsqueeze(-1) : [batch_size, event_num, 1]
        # time_embed : [1, time_feature_size]
        # time_features : [batch_size, event_num, time_feature_size]
        time_features = self.time_embed(time_deltas.unsqueeze(-1))

        # 阶段特征
        # phase_labels ： [batch_size, num_phases]
        # phase_embed embedding : [num_phases, time_feature_size]
        # phase_features : [batch_size, num_phases, time_feature_size]
        # 这里的 num_phases 与 上方的 event_num 数值上是相等的，都是代表有几个操作节点
        phase_features = self.phase_embed(phase_labels)

        # 融合时间特征和阶段特征
        temporal_features = time_features + phase_features

        # 特征拼接
        combined = torch.cat([text_features, temporal_features], dim=1)

        combined = combined.permute(1, 0, 2)  # 调整维度顺序以适应Transformer [L, B, D]
        # 应用掩码，避免填充部分影响注意力计算
        src_key_padding_mask = ~time_mask
        output = self.transformer_encoder(combined, src_key_padding_mask=src_key_padding_mask)

        # 取序列最后时刻
        logits = self.classifier(output[-1, :, :]).squeeze()
        return logits


def train():
    # 示例日志条目
    logs = [
        {
            "events": [
                {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "请提供照片"}
            ],
            "label": 1
        },
        {
            "events": [
                {"operation_time": "2024-05-01 10:00", "operation_type": "进线", "content": "咨询进度"},
                {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
                {"operation_time": "2024-05-01 11:00", "operation_type": "回复", "content": "已处理"}
            ],
            "label": 0
        }
    ]

    dataset = LogisticsDataset(logs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeTextTransformer().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()

            outputs = model(
                batch['time_deltas'].to(device),
                batch['input_ids'].to(device),
                batch['attention_masks'].to(device),
                batch['phase_labels'].to(device),
                batch['time_mask'].to(device)
            )

            loss = criterion(outputs, batch['labels'].to(device))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    print("训练完成")


if __name__ == "__main__":
    # sample_data = [
    #     {
    #         "events": [
    #             {"operation_time": "2024-01-01 10:00", "operation_type": "进线", "content": "包裹破损"},
    #             {"operation_time": "2024-01-02 11:00", "operation_type": "回复", "content": "请提供照片"}
    #         ],
    #         "label": 1
    #     },
    #     {
    #         "events": [
    #             {"operation_time": "2024-05-01 10:00", "operation_type": "进线", "content": "咨询进度"},
    #             {"operation_time": "2024-05-01 11:00", "operation_type": "回复", "content": "已处理"}
    #         ],
    #         "label": 0
    #     }]
    #
    # ot = preprocess_log(sample_data[0], None)
    # print(ot)
    #
    # ot = collate_fn(ot)
    # print(ot)

    train()