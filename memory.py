import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import random, math, torch
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

    @abstractmethod
    def beta_anneal(self):
        pass

    

# Uses sum tree
class ProportionalReplay(AbstractReplay):
    def __init__(self, args):
        self.per_e = args.per_e  
        self.per_a = args.per_a 
        self.per_b = args.per_b
        self.per_b_step = (1 - args.per_b) / args.n_episodes

        self.tree = SumTree(args.memory_size)
        self.new_elem_error = 10000

    def beta_anneal(self):
        self.per_b = np.min([1., self.per_b + self.per_b_step])      

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

        self.per_b = np.min([1., self.per_b + self.per_b_step])

        return batch, np.array(idx_arr), importance_sampling_weight

    def update(self, errors, idxs):
        for e, i in zip(errors, idxs):
            p = self._getPriority(e)
            self.tree.update(i, p)
        
        

class RankReplay(AbstractReplay):
    def __init__(self, args):
        self.new_priority_score = 1.0 
        self.rank_update = args.rank_update
        self.memory_size = args.memory_size

        self.per_a = args.per_a
        self.per_b = args.per_b
        self.per_b_step = (1 - args.per_b) / args.n_episodes
       
        self.id_to_state = {}
        self.buffer = []
        self.seg = [] # Holds idxs from where to sample transitions

        self.counter = 0 # Mapping to dict
        self.steps = 0

    def beta_anneal(self):
        self.per_b = np.min([1., self.per_b + self.per_b_step])        
        
    def size(self):
        return len(self.buffer)

    def add(self, transition):
        # priority = -round(np.random.uniform(0.0, 1.0), 2)
        self.id_to_state[self.counter] = transition
        self.buffer.append((self.new_priority_score, self.counter))
        self.counter += 1

    # From: https://github.com/Wanwannodao/DeepLearning/blob/master/RL/DDQN-PR/pr.py
    def findSegments(self, batch_size):
        n = len(self.buffer)

        dist = np.asarray([(1.0/i)**self.per_a for i in range(1, n)], dtype=np.float32)
        self.p_n = np.sum(dist)
        dist = dist / self.p_n

        cdf = np.cumsum(dist)

        unit = (1.0 - cdf[0])/(batch_size + 1)
        self.seg = [ np.searchsorted(cdf, cdf[0]+unit*i) for i in range(batch_size + 1) ]

    def remove_overfill(self, r):
        keys = [self.buffer[i][1] for i in range(r)]
        del self.buffer[0:r]

        for k in keys:
            del self.id_to_state[k]

   # Stratified sampling
    def get_batch(self, batch_size):
        n = len(self.buffer)

        # These two needs optimization
        if self.steps % self.rank_update == 0:
            self.buffer.sort() # Sorts by priority

            # Remove low priority overfilled items after sorting
            r = len(self.buffer) - self.memory_size
            if r > 1:
                self.remove_overfill(r)
                
            self.findSegments(batch_size) # Has to be every n steps

                
        
        h_indices=[]
        for i in range(batch_size):
            if self.seg[i] != self.seg[i+1]:
                x = np.random.randint(self.seg[i], self.seg[i+1])
            else:
                x = self.seg[i]
            h_indices.append(x)


        batch = [ self.id_to_state[self.buffer[i][1]] for i in h_indices ]

        # IS weights
 
        self.p_n = self.p_n / n
        ranks = np.power(np.array(h_indices) + 1, self.per_a)
        #is_w = np.power(ranks * self.p_n, self.per_b)

        is_w = np.power((ranks * n), -self.per_b)
        is_w /= is_w.max()

        self.steps += 1

        return batch, h_indices, is_w

        # fig = plt.figure()
        # ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(222)

        # ax1.set_title('p_i')
        # ax2.set_title('cdf')

        # plt.ylabel('Priority')
        # plt.xlabel('Rank')
        # for xc in self.seg:
        #     plt.axvline(x=xc)

        # plt.show()

        x = 1
    
    def update(self, errors, idxs):
        for e, i in zip(errors, idxs):
            self.buffer[i] = (e, self.buffer[i][1])

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

    def beta_anneal(self):
        pass



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

    def update(self, errors, idxs):
        self.mem.update(errors, idxs)

    def size(self):
        return self.mem.size()

    def beta_anneal(self):
        self.mem.beta_anneal()
    
