import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        # deque 超过容量后，会自动删除最早的经验。
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 存一条经验。
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 随机抽一批经验，用来训练 DQN。
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer based on proportional TD-error priority."""

    def __init__(self, capacity, alpha=0.6, epsilon=1e-5):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # 新样本先使用当前最大 priority，保证刚进入缓冲区的经验有机会被学习。
        max_priority = self.priorities.max() if self.buffer else 1.0

        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[: len(self.buffer)]
        scaled_priorities = priorities ** self.alpha
        priority_sum = scaled_priorities.sum()
        if priority_sum <= 0 or not np.isfinite(priority_sum):
            sample_probs = np.full(len(self.buffer), 1 / len(self.buffer), dtype=np.float32)
        else:
            sample_probs = scaled_priorities / priority_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)
        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        weights = (len(self.buffer) * sample_probs[indices]) ** (-beta)
        weights = weights / weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        # priority 使用 abs(TD-error)，epsilon 防止 priority 变为 0 后再也采不到。
        for index, td_error in zip(indices, td_errors):
            self.priorities[index] = abs(float(td_error)) + self.epsilon

    def __len__(self):
        return len(self.buffer)
