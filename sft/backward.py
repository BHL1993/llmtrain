import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的前馈神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # 输入层到隐藏层
        self.fc2 = nn.Linear(3, 1)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # 应用sigmoid激活函数
        x = self.fc2(x)
        return x


# 创建模型实例
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.1)  # 学习率为0.1的随机梯度下降

# 创建一些假数据
inputs = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# 训练循环
for epoch in range(1000):  # 进行1000次迭代
    optimizer.zero_grad()  # 清空之前的梯度

    outputs = model(inputs)  # 前向传播
    loss = criterion(outputs, targets)  # 计算损失

    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新权重

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("训练完成")