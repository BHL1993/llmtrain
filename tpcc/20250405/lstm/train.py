from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer

from model import MultiModalEventModel
from DataProcess import EventDataset
from DataProcess import collate_fn
from torch import nn

# 初始化
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = EventDataset("data.json", tokenizer)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

model = MultiModalEventModel(
    num_event_types=3,  # 根据实际类型数量调整
    user_feat_dim=2
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()

# 训练
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")