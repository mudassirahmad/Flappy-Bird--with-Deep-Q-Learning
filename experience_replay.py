from collections import deque
import random

class ExperienceReplay:
    def __init__(self, maxLen, seed=None):
        self.memory = deque(maxlen=maxLen)
        if seed is not None:
            random.seed(seed)

    def append(self,transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)