import torch
from torch import nn


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 输入状态，输出每个动作对应的 Q 值。
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)


if __name__ == "__main__":
    # 简单检查网络输入输出形状是否正确。
    model = DQNNetwork(state_dim=8, action_dim=4)
    dummy_state = torch.zeros(1, 8)
    q_values = model(dummy_state)

    print("Q values shape:", q_values.shape)
    print("Q values:", q_values)
