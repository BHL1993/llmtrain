import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        """
        Args:
            d_model: 模型的总维度（默认512）
            h: 注意力头的数量（默认8）
        """
        super().__init__()
        assert d_model % h == 0, "d_model 必须能被 h 整除"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # 每个头的维度

        # 定义线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)  # 查询矩阵
        self.W_k = nn.Linear(d_model, d_model)  # 键矩阵
        self.W_v = nn.Linear(d_model, d_model)  # 值矩阵
        self.W_o = nn.Linear(d_model, d_model)  # 输出矩阵

    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        Returns:
            输出张量 (batch_size, seq_len, d_model)
            注意力权重 (batch_size, h, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # 线性变换并分割为多个头
        Q = self.W_q(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)  # (B, h, S, d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)  # (B, h, S, d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)  # (B, h, S, d_k)

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, S, S)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, h, S, d_k)

        # 拼接所有头的结果
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # (B, S, d_model)

        # 最终线性变换
        output = self.W_o(context)  # (B, S, d_model)
        return output, attn_weights


# 测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, h=8)
    output, attn = mha(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应该保持 (2, 10, 512)
    print(f"注意力权重形状: {attn.shape}")  # 应该为 (2, 8, 10, 10)