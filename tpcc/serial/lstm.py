import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# 生成示例数据
np.random.seed(42)
event_ids = [1, 2, 3, 4, 5]
actions = ["客服跟进", "升级", "外呼", "转移"]
data = []

for event_id in event_ids:
    for i in range(random.randint(3, 6)):  # 每个事件单3-5条记录
        data.append({
            "event_id": event_id,
            "timestamp": pd.Timestamp(f"2023-01-01 {i}:00:00"),
            "action_type": random.choice(actions),
            "duration": np.random.randint(30, 300),
            "sentiment_score": np.random.uniform(5, 10),
            "label": 1 if random.random() < 0.2 else 0  # 20%的概率产生舆情
        })

df = pd.DataFrame(data)
print("示例数据前5行：")
print(df.head())



# 按事件单ID分组，生成序列数据
max_len = 5  # 最大序列长度

def create_sequences(group):
    group = group.sort_values("timestamp").reset_index(drop=True)
    if len(group) < max_len:
        group = pd.concat([group, pd.DataFrame([{}]*(max_len - len(group)))]).fillna(0)
    else:
        group = group.iloc[:max_len]
    return group

sequences = df.groupby("event_id").apply(create_sequences).reset_index(drop=True)

# 特征编码
categorical_cols = ["action_type"]
for col in categorical_cols:
    le = LabelEncoder()
    sequences[col] = le.fit_transform(sequences[col])

# 特征归一化
numerical_cols = ["duration", "sentiment_score"]
scaler = StandardScaler()
sequences[numerical_cols] = scaler.fit_transform(sequences[numerical_cols])

# 构建输入张量 (样本数, 时间步, 特征数)
X = sequences[["action_type", "duration", "sentiment_score"]].values
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))  # (样本数, 时间步, 特征数)

# 构建标签
y = df.groupby("event_id")["label"].first().values  # 每个事件单最终是否产生舆情
y = y.astype(np.float32)  # 转为float32


class EventDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = EventDataset(X_train, y_train)
test_dataset = EventDataset(X_test, y_test)

# 创建 DataLoader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]    # 取最后一个时间步的输出
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

# 初始化模型
input_dim = 3  # action_type, duration, sentiment_score
hidden_dim = 16
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, output_dim)

# 设备配置（GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 早停逻辑
patience = 5
best_val_loss = float('inf')
counter = 0

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            break