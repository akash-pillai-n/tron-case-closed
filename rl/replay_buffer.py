from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any


@dataclass
class Transition:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    """Simple uniform replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position: int = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)


