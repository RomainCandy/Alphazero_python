import random


class Memory:
    def __init__(self, max_len):
        self.max_len = max_len
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.max_len:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.max_len

    def extend(self, l):
        for elem in l:
            self.push(*elem)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def shuffle(self):
        random.shuffle(self.memory)

    def __getitem__(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return repr(self.memory)
