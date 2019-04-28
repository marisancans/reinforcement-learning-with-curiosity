import numpy as np
import random, math
from collections import deque
import torch
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1


    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return [idx, self.tree[idx], self.data[dataIdx]]



#-------------------- MEMORY --------------------------
# Memory is either deque where batch is uniformly sampled or sumtree with prioritized selection
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    def __init__(self, capacity, is_per):
        if is_per:
            self.tree = SumTree(capacity)
        else:
            self.mem = deque(maxlen=capacity)

        self.is_per = is_per

    def _getPriority(self, error):
        e = (error + self.PER_e) ** self.PER_a
        return e

    def get_entries(self):
        return self.tree.n_entries if self.is_per else len(self.mem)

    def add(self, transition, error=10000):  # because its initially unknown and has to be high priority
        if self.is_per:
            p = self._getPriority(error)
            self.tree.add(p, transition) 
        else:
            self.mem.append(transition)


    def get_batch(self, n):
        return self.uniform_segment_batch(n) if self.is_per else self.random_batch(n)

    def uniform_segment_batch(self, n):
        batch = []
        idx_arr = []
        priority_arr = []
        segment = self.tree.total() / n
        
        if math.isnan(self.tree.total()):
            x = 0

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idx_arr.append(idx)
            priority_arr.append(p)

        sampling_probabilities = priority_arr / self.tree.total()
        importance_sampling_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.PER_b)
        importance_sampling_weight /= importance_sampling_weight.max()

        # state_t = torch.stack([x[0] for x in batch])

        return batch, np.array(idx_arr), importance_sampling_weight

    def random_batch(self, n):
        buffer_size = len(self.mem)
        index = np.random.choice(np.arange(buffer_size), size = n, replace = False)
        batch = [self.mem[i] for i in index]

        return batch, None, None

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)