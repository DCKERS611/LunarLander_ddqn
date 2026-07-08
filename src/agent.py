import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from networks import DQNNetwork, DuelingDQNNetwork
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgent:
    """统一管理 DQN、Double DQN 和 Dueling Double DQN 的智能体。"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=64,
        double_dqn=False,
        dueling=False,
        prioritized_replay=False,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=100_000,
        device=None,
    ):
        # action_dim 用于 epsilon-greedy 随机探索时确定动作数量。
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_steps = 0

        # double_dqn 控制目标值计算方式，dueling 控制使用哪一种网络结构。
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.prioritized_replay = prioritized_replay
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # online network 负责当前动作价值估计；target network 只定期同步，用来稳定目标值。
        network_class = DuelingDQNNetwork if dueling else DQNNetwork
        self.q_net = network_class(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = network_class(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, epsilon):
        # epsilon-greedy：训练前期更多随机探索，后期更多利用网络学到的策略。
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        # Gymnasium 返回的是 numpy 数组，这里转成 PyTorch 张量并增加 batch 维度。
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        # 每一步交互得到的经验都会先进入回放池，再从回放池随机采样训练。
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        # 回放池样本不足一个 batch 时先不训练，避免采样报错。
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.update_steps += 1

        # 从经验回放池采样一批 transition，打破连续时间步之间的相关性。
        if self.prioritized_replay:
            beta = self.current_per_beta()
            sample = self.replay_buffer.sample(self.batch_size, beta=beta)
            states, actions, rewards, next_states, dones, indices, weights = sample
            weights = torch.tensor(weights, device=self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            indices = None
            weights = None

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        # 当前网络估计：Q(s, a)。
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN：online network 选动作，target network 评估该动作价值。
                # 这样可以缓解普通 DQN 中 max 操作带来的 Q 值过估计问题。
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_q_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                # DQN：target network 直接取下一状态最大 Q 值。
                next_q_values = self.target_q_net(next_states).max(dim=1).values

            # done=1 表示 episode 已结束，终止状态后面不再叠加未来奖励。
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        td_errors = target_q_values - q_values

        # Huber loss 对异常 TD error 更稳；PER 模式下额外乘以 importance-sampling weight。
        if self.prioritized_replay:
            loss_elements = F.smooth_l1_loss(q_values, target_q_values, reduction="none")
            loss = (weights * loss_elements).mean()
        else:
            loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized_replay:
            td_errors_np = td_errors.detach().abs().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors_np)

        return loss.item()

    def current_per_beta(self):
        # beta 从 per_beta_start 逐步退火到 1.0，训练后期更充分校正优先采样偏差。
        progress = min(1.0, self.update_steps / self.per_beta_frames)
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

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
