import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.torch_utils import init_parameters
from modules.torch_utils import to_numpy
from modules.state_utils import average

import random, gym, os, cv2, time, re, gc
from gym import envs
import numpy as np
from sum_tree import SumTree, Memory

from collections import deque
from encoder import FeatureExtractor
from agent_dqn import AgentDQN

class AgentCurious(AgentDQN):
    def __init__(self, args, name):
        super().__init__(args, name)


  

    def after_step(self, act_values, reward, next_state, is_terminal):
        if self.args.is_images:
            if self.current_step % self.args.n_frame_skip == 0: # Frame skipping, only every n-th frame is taken
                next_state = self.get_next_sequence(next_state, not is_terminal)
            else: 
                return 
        else:
            next_state = self.encode_state(next_state) # TODO - This encodes each state, its slow

        transition = [self.current_state, act_values, reward, next_state, is_terminal]
        self.memory.add(transition)   

        self.current_state = next_state

    def terminal_episode(self):
        super(AgentCurious, self).terminal_episode()

        

    def remember_episode(self,):
        #if self.args.has_images:
        loss_dqn = loss_dqn.detach().mean()
            
        super(AgentCurious, self).remember_episode(loss_dqn)

       

    def replay(self, is_terminal):     
        minibatch, idxs, importance_sampling_weight = self.memory.get_batch(self.args.batch_size)

        state_t = torch.stack(tuple(minibatch[:, 0]))
        recorded_action = np.stack(minibatch[:, 1])
        reward = np.stack(minibatch[:, 2])
        next_state_t = torch.stack(tuple(minibatch[:, 3]))
        done = np.stack(minibatch[:, 4])


       
 
        loss_dqn = self.train_dqn_model(state_t, recorded_action, reward, next_state_t, done, importance_sampling_weight, idxs)



        self.remember_episode(loss_dqn, loss_cos, loss_inv, loss)
        self.backprop(loss, is_terminal)

