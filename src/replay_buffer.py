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