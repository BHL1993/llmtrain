import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torch.distributions import Categorical
import numpy as np


# 定义策略网络（Actor-Critic结构）
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)  # 输出动作概率
        self.critic = nn.Linear(64, 1)  # 输出状态价值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value


# 自定义奖励函数
def custom_reward_function(state, done):
    """
    输入: 
        state (array): 当前状态 [车位置, 车速, 杆角度, 杆角速度]
        done (bool): 是否结束
    返回:
        reward (float): 自定义奖励值
    """
    _, _, angle, _ = state  # 杆角度（弧度）

    # 基础奖励：每步存活 +1
    reward = 1.0

    # 惩罚项：杆角度偏离越大，惩罚越大（绝对值角度接近0最优）
    angle_penalty = -0.1 * abs(angle)
    reward += angle_penalty

    # 如果游戏结束，额外惩罚
    if done:
        reward -= 10.0
    return reward


# PPO超参数
GAMMA = 0.99
CLIP_EPSILON = 0.2
EPOCHS = 4
BATCH_SIZE = 64
LR = 3e-4

# 初始化环境和模型
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy = Policy(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=LR)


# PPO训练循环
def train():
    state = env.reset()
    episode_reward = 0
    saved_log_probs = []
    saved_values = []
    saved_rewards = []
    saved_states = []
    saved_dones = []

    # 收集一个批次的轨迹数据
    for _ in range(BATCH_SIZE):
        state_tensor = torch.FloatTensor(state)
        action_probs, value = policy(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()

        next_state, _, done, _ = env.step(action.item())

        # 使用自定义奖励函数（替代RM或环境默认奖励）
        reward = custom_reward_function(next_state, done)

        saved_log_probs.append(m.log_prob(action))
        saved_values.append(value)
        saved_rewards.append(reward)
        saved_states.append(state_tensor)
        saved_dones.append(done)

        state = next_state if not done else env.reset()
        episode_reward += reward

    # 计算折扣回报和优势
    returns = []
    advantages = []
    R = 0
    for r, done in zip(reversed(saved_rewards), reversed(saved_dones)):
        R = r + GAMMA * R * (1 - done)
        returns.insert(0, R)
    returns = torch.tensor(returns)
    advantages = returns - torch.stack(saved_values).squeeze()

    # PPO优化步骤
    for _ in range(EPOCHS):
        for idx in range(BATCH_SIZE):
            log_prob, value = policy(saved_states[idx])
            ratio = torch.exp(log_prob[saved_actions[idx]] - saved_log_probs[idx])
            clip_ratio = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            policy_loss = -torch.min(ratio * advantages[idx], clip_ratio * advantages[idx])
            value_loss = 0.5 * (value - returns[idx]) ** 2
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return episode_reward


# 训练主循环
for episode in range(1000):
    reward = train()
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {reward:.1f}")