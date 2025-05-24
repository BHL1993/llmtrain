import torch
import torch.nn as nn
import math


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model=1536, num_heads=12, groups=6):
        """
        Args:
            d_model: 模型维度 (默认 512)
            num_heads: 总查询头数 (h)
            groups: 分组数量 (g), 必须满足 num_heads % groups == 0
        """
        super().__init__()
        assert num_heads % groups == 0, "num_heads 必须能被 groups 整除"

        self.d_model = d_model
        self.num_heads = num_heads  # 总查询头数 (h)
        self.groups = groups  # 分组数量 (g)
        self.heads_per_group = num_heads // groups  # 每组的查询头数 (k)
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)  # 查询投影 (h 个头)
        self.W_k = nn.Linear(d_model, d_model // (num_heads // groups))  # 键投影 (g 个头)
        self.W_v = nn.Linear(d_model, d_model // (num_heads // groups))  # 值投影 (g 个头)
        self.W_o = nn.Linear(d_model, d_model)  # 输出投影

    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        Returns:
            输出张量 (batch_size, seq_len, d_model)
            注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()
        h, g, k = self.num_heads, self.groups, self.heads_per_group

        # 生成 Q/K/V
        Q = self.W_q(x).view(batch_size, seq_len, h, self.d_k).transpose(1, 2)  # (B, h, S, d_k)
        K = self.W_k(x).view(batch_size, seq_len, g, self.d_k).transpose(1, 2)  # (B, g, S, d_k)
        V = self.W_v(x).view(batch_size, seq_len, g, self.d_k).transpose(1, 2)  # (B, g, S, d_k)

        # 将 K 和 V 复制到每组中的 k 个查询头
        # 操作后维度: (B, g, k, S, d_k) → (B, h, S, d_k)
        K = K.unsqueeze(2).expand(-1, -1, k, -1, -1).contiguous().view(batch_size, h, seq_len, self.d_k)
        V = V.unsqueeze(2).expand(-1, -1, k, -1, -1).contiguous().view(batch_size, h, seq_len, self.d_k)

        # 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, S, S)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, h, S, d_k)

        # 合并多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # (B, S, d_model)
        output = self.W_o(context)
        return output, attn_weights


# 测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    x = torch.randn(batch_size, seq_len, d_model)

    gqa = GroupedQueryAttention()
    output, attn = gqa(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应保持 (2, 10, 512)
    print(f"注意力权重形状: {attn.shape}")  # 应为 (2, 8, 10, 10)