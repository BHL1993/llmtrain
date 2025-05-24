import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# 配置参数
class Config:
    vocab_size = 10  # 操作类型总数（0-9为常规操作，10为特殊舆情标记）
    max_len = 6  # 最大序列长度（根据业务场景调整）
    d_model = 32
    nhead = 2
    num_layers = 2
    batch_size = 8
    num_epochs = 15
    learning_rate = 1e-3


# 生成模拟数据（实际使用时需替换为真实数据）
def generate_mock_data(num_samples=200):
    # 生成操作序列和标签
    sequences = []
    labels = []

    # 正样本：包含舆情风险的序列模式
    for _ in range(num_samples // 2):
        # 风险模式示例：多次转移后升级失败
        seq = np.random.choice([1, 3, 5], size=np.random.randint(3, 5))  # 常规操作
        seq = np.append(seq, [7, 9]) if np.random.rand() > 0.5 else seq  # 添加风险操作组合
        sequences.append(seq.tolist())
        labels.append(1)  # 发生舆情

    # 负样本：正常处理流程
    for _ in range(num_samples // 2):
        seq = np.random.choice([2, 4, 6, 8], size=np.random.randint(2, 6))
        sequences.append(seq.tolist())
        labels.append(0)  # 未发生舆情

    return sequences, labels


# 自定义数据集
class ServiceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 填充/截断
        padded_seq = np.zeros(Config.max_len, dtype=np.int64)
        valid_len = min(len(seq), Config.max_len)
        padded_seq[:valid_len] = seq[:valid_len]

        # 注意力掩码（1表示有效位置）
        attention_mask = np.zeros(Config.max_len)
        attention_mask[:valid_len] = 1

        return {
            'input_ids': torch.LongTensor(padded_seq),
            'attention_mask': torch.BoolTensor(attention_mask.astype(bool)),
            'label': torch.FloatTensor([label])
        }


# Transformer分类模型
class RiskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.vocab_size, Config.d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, Config.max_len, Config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.d_model,
            nhead=Config.nhead,
            dropout=0.1,
            batch_first=True  # 使用更直观的batch_first格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=Config.num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(Config.d_model, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        # 嵌入层 + 位置编码
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer编码（自动处理mask）
        x = self.encoder(x, src_key_padding_mask=~attention_mask)

        # 取最后一个有效位置的输出
        last_idx = attention_mask.sum(dim=1) - 1
        last_output = x[torch.arange(x.size(0)), last_idx]

        # 分类
        return self.classifier(last_output).squeeze()


device = "cpu"
# 评估函数
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs, masks)
            loss = nn.BCELoss()(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc, f1


# 训练流程
def train():
    # 生成数据
    sequences, labels = generate_mock_data(500)
    train_seq, test_seq, train_lb, test_lb = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )

    # 创建DataLoader
    train_dataset = ServiceDataset(train_seq, train_lb)
    test_dataset = ServiceDataset(test_seq, test_lb)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RiskPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # 训练循环
    best_f1 = 0
    for epoch in range(Config.num_epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = nn.BCELoss()(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 评估
        train_loss, train_acc, train_f1 = evaluate(model, train_loader)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader)

        # 保存最佳模型
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch + 1:02d}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Test  Loss: {test_loss:.4f}  | Acc: {test_acc:.4f} | F1: {test_f1:.4f}\n")


if __name__ == "__main__":
    train()