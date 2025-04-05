from torch.utils.data import Dataset
import json
import torch

# {
#     # 事件序列特征
#     "events": [
#         {
#             "type": tensor(int),  # 事件类型ID (如进线=0)
#             "text": {
#                 "input_ids": tensor(int),  # BERT输入ID
#                 "attention_mask": tensor(int)
#             },
#             "time_intervals": tensor(float)  # [距离上次间隔, 距离首次间隔]
#         },
#         # ... 更多事件
#     ],
#
#     # 用户特征
#     "user_features": tensor(float),  # [历史进线次数, 历史舆情次数]
#
#     # 标签
#     "label": tensor(float)  # 是否下一个事件是舆情
# }

class EventDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_seq_length=128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        with open(json_path) as f:
            self.data = json.load(f)

        # 构建事件类型映射表
        self.event_types = {"进线": 0, "跟进": 1, "舆情": 2}  # 根据实际数据扩展

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 处理每个事件
        events = []
        for event in sample["事件操作序列"]:
            # 文本编码
            text_enc = self.tokenizer(
                event["沟通文本"],
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )

            # 时间间隔 (小时为单位)
            time_intervals = torch.tensor([
                event["距离上一次事件操作的时间间隔"],
                event["距离第一次事件操作的时间间隔"]
            ], dtype=torch.float)

            events.append({
                "type": torch.tensor(self.event_types[event["类型"]], dtype=torch.long),
                "text": text_enc,
                "time_intervals": time_intervals
            })

        # 用户特征
        user_feat = torch.tensor([
            sample["用户历史进线次数"],
            sample["历史舆情次数"]
        ], dtype=torch.float)

        # 标签 (最后一个事件是否为舆情)
        label = 1.0 if events[-1]["类型"] == "舆情" else 0.0

        return {
            "events": events,
            "user_features": user_feat,
            "label": torch.tensor(label, dtype=torch.float)
        }


from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # 填充事件序列
    padded_events = []
    max_seq_len = max(len(sample['events']) for sample in batch)

    for sample in batch:
        events = sample['events']
    # 填充到最大长度
    padded = []
    for _ in range(max_seq_len - len(events)):
        padded.append({
            "type": torch.tensor(0),  # 用0填充
            "text": {"input_ids": torch.zeros(max_seq_length),
                     "attention_mask": torch.zeros(max_seq_length)},
            "time_intervals": torch.zeros(2)
        })
    padded_events.append(events + padded)

    # 重组数据结构
    return {
        "events": padded_events,
        "user_features": torch.stack([s['user_features'] for s in batch]),
        "label": torch.stack([s['label'] for s in batch])
    }