import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

def _time_to_phase(hours):
    """阶段划分函数"""
    if hours < 24:
        return 0
    elif hours < 72:
        return 1
    else:
        return 2


def _process_log(log_entries):
    # 时间特征计算
    start_time = pd.to_datetime(log_entries[0]["操作时间"])
    time_deltas, phase_labels, texts = [], [], []
    for entry in log_entries:
        curr_time = pd.to_datetime(entry["操作时间"])
        delta_hours = (curr_time - start_time).total_seconds() / 3600
        time_deltas.append(delta_hours)
        phase_labels.append(_time_to_phase(delta_hours))
        texts.append(f"{entry['操作类型']}：{entry['内容概述']}")

    # 标签生成（假设最后一个步骤是否包含投诉）
    label = 1 if any("投诉" in t for t in texts) else 0

    return {
        "time_deltas": torch.FloatTensor(time_deltas),
        "phase_labels": torch.LongTensor(phase_labels),
        "texts": texts,
        "labels": torch.FloatTensor([label])
    }


def _process_log1(log_entries):
    texts = [f"{entry['操作类型']}：{entry['内容概述']}" for entry in log_entries]
    label = 1 if any("投诉" in t for t in texts) else 0

    full_text = " | ".join(texts)  # 用分隔符连接多步骤

    delta_hours = (pd.to_datetime(log_entries[-1]["操作时间"]) -
                       pd.to_datetime(log_entries[0]["操作时间"])).total_seconds() / 3600
    phase = _time_to_phase(delta_hours)

    return {
        "time_deltas": torch.FloatTensor([delta_hours]),
        "phase_labels": torch.LongTensor([phase]),
        "full_text": full_text,
        "labels": torch.FloatTensor([label])
    }


def reorganize_events(sequence):
    """
    将原始事件序列转换为LLM友好格式
    """
    template = [
        "时间：{time}",
        "操作类型：{action}",
        "内容摘要：{content}"
    ]

    structured_events = []
    for event in sequence:
        event_str = "\n".join([
            t.format(
                time=event['操作时间'],
                action=event['操作类型'],
                content=event['内容概述']
            ) for t in template
        ])
        structured_events.append(f"【事件记录】\n{event_str}")

    return "\n\n".join(structured_events)

if __name__ == "__main__":
    # 示例数据
    # sample_logs = [
    #     [
    #         {"操作时间": "2024-10-10 10:10:10", "操作类型": "客户进线", "内容概述": "快递破损"},
    #         {"操作时间": "2024-10-13 11:10:10", "操作类型": "投诉", "内容概述": "未解决"}
    #     ],
    #     [
    #         {"操作时间": "2024-10-11 09:00:00", "操作类型": "客户进线", "内容概述": "包裹未送达"},
    #         {"操作时间": "2024-10-11 12:00:00", "操作类型": "处理完成", "内容概述": "已重新派送"}
    #     ]
    # ]
    #
    # out = _process_log(sample_logs[0])
    # out1 = _process_log1(sample_logs[0])
    # print(out)

    # logs = [
    #         {"操作时间": "2024-10-10 10:10:10", "操作类型": "客户进线", "内容概述": "快递破损"},
    #         {"操作时间": "2024-10-13 11:10:10", "操作类型": "投诉", "内容概述": "未解决"}
    #     ]
    #
    # ou = reorganize_events(logs)
    # print(ou)

    # x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(x)  # 输出: torch.Size([2, 3])
    #
    # x_unsqueezed = x.unsqueeze(0)
    # print(x_unsqueezed)  # 输出: torch.Size([1, 2, 3])
    #
    # x_unsqueezed_last = x.unsqueeze(-1)  # -1表示最后一个维度的位置
    # print(x_unsqueezed_last)  # 输出: torch.Size([2, 3, 1])
    #
    # import torch
    # import torch.nn as nn
    #
    # # 假设输入张量的形状为 (1, 3, 4, 4)
    # input_tensor = torch.randn(1, 3, 4, 4)
    # print("输入张量的形状:", input_tensor.shape)
    #
    # # 创建一个 Flatten 层，从第 2 维开始展平
    # flatten = nn.Flatten(start_dim=1)
    #
    # # 使用 Flatten 层
    # output_tensor = flatten(input_tensor)
    # print("展平后的张量形状:", output_tensor.shape)
    # print(input_tensor)
    # print(output_tensor)

    # import torch
    #
    # # 创建一个3维张量
    # time_emb = torch.randn(4, 5, 6)
    # print("Original shape:", time_emb.shape)
    #
    # # 调整维度顺序
    # time_emb_permuted = time_emb.permute(1, 0, 2)
    # print("Permuted shape:", time_emb_permuted.shape)

    base_model_name = "/Users/baihailong/PycharmProjects/train/local_model/llama-7b"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    #text_encoder = AutoModelForCausalLM.from_pretrained(base_model_name)
    print('')

