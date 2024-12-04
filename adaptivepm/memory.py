from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class PortfolioVectorMemory:
    """Implements the Portfolio Vector Memory inspired by the idea of experience replay memory (Mnih et al., 2013),
    see pg. 13-14 of paper
    A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
    """

    n_samples: int
    m_noncash_assets: int
    initial_weight: Optional[torch.tensor] = None
    memory: torch.tensor = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.memory = torch.ones(self.n_samples, self.m_noncash_assets) / (
            self.m_noncash_assets + 1
        )
        self.memory = self.memory.to(self.device)

    def update_memory_stack(self, new_weights: torch.tensor, indices: torch.tensor):
        self.memory[indices] = new_weights

    def get_memory_stack(self, indices):
        return self.memory[indices]


@dataclass
class ExperienceReplayMemory:
    """Implements the Experience Replay Memory by (Mnih et al. 2013)
    c.f https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    by storing the experience (Xt, w(t-1), wt, rt, X(t+1), done)"""

    n_samples: int
    buffer: deque = field(init=False)

    def __post_init__(self):
        self.buffer = deque()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
        return (
            torch.stack(state),
            torch.stack(action),
            torch.tensor(reward, dtype=torch.float32),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float32),
        )
