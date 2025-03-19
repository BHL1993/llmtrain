#

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


# 定义多模态模型
class LogisticsRiskModel(nn.Module):
    def __init__(self, bert_path='bert-base-chinese', stats_dim=5, time_dim=1, hidden_dim=768):
        super().__init__()
        # 文本编码器
        self.bert = BertModel.from_pretrained(bert_path)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        # 时间间隔编码器（事件级）
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim))

        # 全局统计编码器（用户级）
        self.global_encoder = nn.Sequential(
            nn.Linear(stats_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim))

        # 交叉注意力（全局特征→事件特征）
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # 自注意力（事件序列建模）
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, input_ids, attention_mask, time_features, global_stats):
        # 文本编码 (batch_size, seq_len, hidden_dim)
        text_outputs = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        text_features = self.text_proj(text_outputs[:, 0, :].unsqueeze(1))  # 取CLS向量

        # 时间编码 (batch_size, seq_len, hidden_dim)
        time_features = self.time_encoder(time_features.unsqueeze(-1))

        # 拼接事件级特征 (batch_size, seq_len, hidden_dim*2)
        event_features = torch.cat([text_features, time_features], dim=-1)

        # 全局统计编码 (batch_size, hidden_dim)
        global_features = self.global_encoder(global_stats).unsqueeze(1)  # (batch, 1, hidden)

        # 交叉注意力：事件特征作为Query，全局特征作为Key/Value
        cross_output, _ = self.cross_attn(
            query=event_features,
            key=global_features.expand(-1, event_features.size(1), -1),
            value=global_features.expand(-1, event_features.size(1), -1))

        # 自注意力
        self_output, _ = self.self_attn(cross_output, cross_output, cross_output)

        # 分类预测
        logits = self.classifier(self_output.mean(dim=1))
        return logits


from torch.utils.data import Dataset, DataLoader
import pandas as pd


# 自定义数据集
class LogisticsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.data = pd.read_json(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # 文本处理
        texts = [event["内容概述"] for event in sample["操作记录"]]
        encoding = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        # 时间间隔特征 (假设已预处理为数值)
        time_features = torch.tensor(sample["时间间隔"], dtype=torch.float)  # (seq_len,)
        # 全局统计特征
        global_stats = torch.tensor([sample["月投诉次数"], ...], dtype=torch.float)  # (stats_dim,)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'time_features': time_features,
            'global_stats': global_stats,
            'label': torch.tensor(sample["label"], dtype=torch.long)
        }


# 初始化组件
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = LogisticsDataset("data.json", tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
model = LogisticsRiskModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(10):
    model.train()
    for batch in dataloader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            time_features=batch['time_features'],
            global_stats=batch['global_stats'])
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # 验证步骤略...


# 注意力权重可视化
def plot_attention(attn_weights, event_texts):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attn_weights.mean(dim=0), annot=True,
                xticklabels=event_texts, yticklabels=event_texts)
    plt.title("事件间注意力权重")
    plt.show()

# SHAP全局特征重要性
def shap_analysis(model, sample):
    explainer = shap.DeepExplainer(
        model,
        torch.zeros(1, sample['input_ids'].size(1)))  # 背景样本
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, feature_names=["投诉次数", "历史处理时效", ...])

# 词级贡献分析 (使用Captum)
from captum.attr import LayerIntegratedGradients

def token_attribution(model, sample):
    lig = LayerIntegratedGradients(model, model.bert.embeddings)
    attributions, delta = lig.attribute(
        inputs=sample['input_ids'].unsqueeze(0),
        target=1,
        return_convergence_delta=True)
    # 可视化
    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
    plt.bar(range(len(tokens)), attributions.squeeze().detach().numpy())
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title("文本词级贡献度")
    plt.show()

# 使用示例
sample = dataset[0]
with torch.no_grad():
    attn_weights = model.self_attn_weights  # 需修改模型以返回注意力权重
plot_attention(attn_weights, ["进线描述", "回访记录"])
shap_analysis(model, sample)
token_attribution(model, sample)