import torch
from torch import nn


class DQNNetwork(nn.Module):
    """基础 DQN 网络：直接从状态估计每个离散动作的 Q 值。"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # LunarLander 的状态是低维向量，因此使用全连接网络即可。
        # 网络最后一层不加激活函数，因为 Q 值可以是任意实数。
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN 网络：将 Q 值拆成状态价值 V(s) 和动作优势 A(s, a)。"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 共享特征层：先把原始状态映射为高维特征，供两个分支共同使用。
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value 分支：估计当前状态本身好不好，输出一个标量 V(s)。
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Advantage 分支：估计每个动作相对平均动作的优势，输出 action_dim 个值。
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 合并公式：Q(s,a) = V(s) + A(s,a) - mean(A(s,a))。
        # 减去均值可以避免 V 和 A 的分解不唯一，使训练更稳定。
        return value + advantage - advantage.mean(dim=1, keepdim=True)


if __name__ == "__main__":
    # 简单检查网络输入输出形状是否正确。
    model = DQNNetwork(state_dim=8, action_dim=4)
    dueling_model = DuelingDQNNetwork(state_dim=8, action_dim=4)
    dummy_state = torch.zeros(1, 8)
    q_values = model(dummy_state)
    dueling_q_values = dueling_model(dummy_state)

    print("DQN Q values shape:", q_values.shape)
    print("Dueling DQN Q values shape:", dueling_q_values.shape)
