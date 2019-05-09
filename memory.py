import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import random, math, torch, heapq
from abc import ABC, abstractmethod

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

class AbstractReplay(ABC):
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def add(self, transition):
        pass

    @abstractmethod
    def get_batch(self, n):
        pass

    

# Uses sum tree
class ProportionalReplay(AbstractReplay):
    def __init__(self, args):
        self.per_e = args.per_e  
        self.per_a = args.per_a 
        self.per_b = args.per_b
        self.per_b_annealing = args.per_b_annealing

        self.tree = SumTree(args.memory_size)
        self.new_elem_error = 10000

    def _getPriority(self, error):
        e = (error + self.per_e) ** self.per_a
        return e

    def size(self):
        return self.tree.n_entries

    def add(self, transition):
        p = self._getPriority(self.new_elem_error)
        self.tree.add(p, transition) 

    def get_batch(self, n):
        batch = []
        idx_arr = []
        priority_arr = []
        segment = self.tree.total() / n
        
        if math.isnan(self.tree.total()):
            x = 0 # Testing purpose

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idx_arr.append(idx)
            priority_arr.append(p)

        sampling_probabilities = priority_arr / self.tree.total()
        importance_sampling_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.per_b)
        importance_sampling_weight /= importance_sampling_weight.max()

        self.per_b = np.min([1., self.per_b + self.per_b_annealing])

        return batch, np.array(idx_arr), importance_sampling_weight

    def update(self, error, idx):
        p = self._getPriority(error)
        self.tree.update(idx, p)

# Uses min binary heap
# States are stored in list, heap operates only with indexes so that values arent copied when using heapsort()
class RankReplay(AbstractReplay):
    def __init__(self, args):
        self.new_elem_error = -1.0 # has to be negative, because min heap is used 
        self.per_a = args.per_a
        self.memory_size = args.memory_size
        self.id_to_state = {}
        self.heap = []
        self.counter = 0 # Counts how many items have been inserted into heap, used as secondary key
        
    def size(self):
        return len(self.heap)

    def add(self, transition):
        priority = round(np.random.uniform(0.0, 1.0), 2)
        t = (priority, self.counter)
        self.id_to_state[self.counter] = transition

        # Check if heap if full, if so, smallest priority will be replaced
        if len(self.heap) < self.memory_size:
            heapq.heappush(self.heap, t)
        else:
            spilled_value = heapq.heappushpop(self.heap, t) # TODO ------ REMOVE OLD VALUES FROM id_to_state

        self.counter += 1

            
    def heapsort(self):
        return [heapq.heappop(self.heap) for _ in range(len(self.heap))]

 
    def get_batch(self, k):
        x = heapq.heappop(self.heap)
        x = self.heapsort()

        n = len(x)

        dist = np.asarray([(1.0/(i))**self.per_a for i in range(1, n)], dtype=np.float32)
        self.p_n = np.sum(dist)
        dist = dist / self.p_n

        # for IS weights
        self.p_n = self.p_n / n

        # cumulative distibution
        cdf = np.cumsum(dist)

        unit = (1.0 - cdf[0])/k
        # comprehension is faster
        self.seg = [ np.searchsorted(cdf, cdf[0]+unit*i) for i in range(k) ]

        # # searchsorted only works on ascending, so we need to reverse
        # segments = []
        # for s in range(1, n):
        #     idx = d.size - np.searchsorted(d[::-1], step * s, side = "right")
        #     segments.append(idx)
        # segments = [ for s in range(n)]

        d = [i[0] for i in x]

        x = 1
        plt.plot(d)
        plt.ylabel('Priority')
        plt.xlabel('Rank')
        for xc in self.seg:
            plt.axvline(x=xc)

        plt.show()

        x = 1

class RandomReplay(AbstractReplay):
    def __init__(self, args):
        self.buffer = deque(maxlen=args.memory_size)
    
    def size(self):
        return len(self.buffer)

    def add(self, transition):
        self.buffer.append(transition)

    def get_batch(self, n):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size = n, replace = False)
        batch = [self.buffer[i] for i in index]

        return batch, None, None



#-------------------- MEMORY --------------------------
# Wrapper to 3 types of memory replay - random uniform, proportional, rank
class Memory:  
    def __init__(self, args):
        replay_types = {
            'random': 0,
            'proportional': 1,
            'rank': 2
        }

        self.curr_type = replay_types[args.prioritized_type]
            
        mem_types = {
            0: RandomReplay(args),
            1: ProportionalReplay(args),
            2: RankReplay(args)
        }
        
        self.mem = mem_types[self.curr_type] 

    def add(self, transition):  
        self.mem.add(transition)

    def get_batch(self, n):
        return self.mem.get_batch(n)

    def update(self, error, idx):
        self.mem.update(error, idx)

    def size(self):
        return self.mem.size()
    
