import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from networks import DQNNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        device=None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, epsilon):
        # epsilon-greedy：有概率随机探索，否则选择 Q 值最大的动作。
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state = np.array(state, dtype=np.float32)
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        # 当前网络估计：Q(s, a)。
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 目标网络估计：r + gamma * max Q(s', a')。
            next_q_values = self.target_q_net(next_states).max(dim=1).values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        # 定期同步目标网络，让训练更稳定。
        self.target_q_net.load_state_dict(self.q_net.state_dict())


if __name__ == "__main__":
    agent = DQNAgent(state_dim=8, action_dim=4)

    for _ in range(100):
        state = np.zeros(8, dtype=np.float32)
        next_state = np.ones(8, dtype=np.float32)
        agent.store_transition(state, 0, 1.0, next_state, False)

    loss = agent.update()
    print("Device:", agent.device)
    print("Loss:", loss)
