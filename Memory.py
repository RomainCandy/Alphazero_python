import random


class Memory:
    def __init__(self, max_len):
        self.max_len = max_len
        self.memory = []
        self.position = 0

    def push(self, state, player_turn, target_pi, target_v=None):
        if len(self.memory) < self.max_len:
            self.memory.append(None)
        self.memory[self.position] = (state, player_turn, target_pi, target_v)
        self.position = (self.position + 1) % self.max_len

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
