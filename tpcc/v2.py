# 1、动态阶段调整
#   将阶段阈值设为可训练参数，模型自动学习最优划分点
#   初始化使用业务经验值，加速收敛
# 2、可微分阶段划分
# phase1_prob = torch.sigmoid(self.thresholds[0] - hours)
# phase2_prob = torch.sigmoid(self.thresholds[1] - hours)
# 利用sigmoid函数生成软阶段分配概率
# 保留梯度传播路径，使阈值可学习
# 3、直通估计器
# hard_assign = F.gumbel_softmax(..., hard=True)
# soft_assign = F.softmax(...)
# return hard_assign + (soft_assign - soft_assign.detach())
# 前向传播使用硬分配保证阶段离散性
# 反向传播用软概率保持梯度可传
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPhaseModel(nn.Module):
    def __init__(self, init_thresholds=[24.0, 72.0]):
        super().__init__()
        # 可学习的阶段阈值参数
        self.thresholds = nn.Parameter(torch.tensor(init_thresholds, dtype=torch.float32))

        # 阶段嵌入层（保持3个阶段）
        self.phase_embed = nn.Embedding(3, 128)

        # 时间编码层
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )

        # 梯度直通估计器
        self.ste = StraightThroughEstimator()

    def _dynamic_phase(self, hours):
        """可微分的动态阶段划分"""
        # 计算软阶段分配
        phase1_prob = torch.sigmoid(self.thresholds[0] - hours)  # 小于阈值1的概率
        phase2_prob = torch.sigmoid(self.thresholds[1] - hours)  # 小于阈值2的概率

        # 生成软分配矩阵
        soft_assign = torch.stack([
            phase1_prob,  # 阶段0的概率
            phase2_prob - phase1_prob,  # 阶段1的概率
            1 - phase2_prob  # 阶段2的概率
        ], dim=-1)

        # 应用直通估计器
        hard_assign = self.ste(soft_assign)
        return hard_assign

    def forward(self, time_features):
        """
        time_features: 时间差值张量 [batch_size, seq_len]
        """
        # 时间编码
        time_emb = self.time_encoder(time_features.unsqueeze(-1))  # [B, L, 128]

        # 动态阶段分配
        phase_probs = self._dynamic_phase(time_features)  # [B, L, 3]

        # 融合阶段特征
        phase_emb = torch.einsum('blp,ep->ble', phase_probs, self.phase_embed.weight)

        # 组合特征
        combined = time_emb + phase_emb
        return combined


class StraightThroughEstimator(nn.Module):
    """直通估计器实现"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 前向传播使用硬分配
        hard_assign = F.gumbel_softmax(x, tau=1.0, hard=True)

        # 反向传播使用软概率
        soft_assign = F.softmax(x, dim=-1)
        return hard_assign + (soft_assign - soft_assign.detach())