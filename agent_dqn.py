import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.torch_utils import init_parameters
from modules.torch_utils import to_numpy
from modules.state_utils import average

import random, gym, os, cv2, time
from gym import envs
import numpy as np
from collections import deque
from sum_tree import SumTree, Memory

from collections import deque


class AgentDQN(nn.Module):
    def __init__(self, args, name):
        super().__init__()

        # --------- AGENT ---------------
        self.name = name
        self.check_base_args(args)
        self.args = args

        # --------- ENVIROMENT ----------
        self.env = gym.make(self.args.env_name)
        self.current_state = None # Gets set in self.reset_env()

        # --------- ENV STATE --------------- 
        self.n_states = self.env.observation_space.shape[0]

        self.state_max_val = self.env.observation_space.low.min()
        self.state_min_val = self.env.observation_space.high.max()
        self.n_actions = self.env.action_space.n
        self.epsilon = 1.0
        self.epsilon_start = 1.0

        # --------- MODELS --------------
        self.dqn_model = self.build_dqn_model().to(self.args.device)
        self.dqn_model = init_parameters('dqn', self.dqn_model)

        self.target_model = self.build_dqn_model().to(self.args.device)

        self.dqn_model_loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.dqn_model.parameters(), lr = self.args.learning_rate)

        # --------- INTERNAL STATE -------
        self.current_episode = 0
        self.total_steps = 0 # in all episodes combined
        self.current_step = 0 # only in current episode
        self.memory = Memory(capacity=self.args.memory_size, is_per=self.args.has_prioritized)

        # ----- TRAINING BUFFER --------
        self.loss_dqn = [0]
        self.ers = [0]
        
        # ----- EPISODE BUFFER  --------
        self.e_loss_dqn = []
        self.e_reward = 0

        self.update_target()

    def check_base_args(self, args):
        if args.batch_size < 4:
            print('Batch size too small!')
            os._exit(0)

        self.check_args(args)

    # OVERRIDABLE
    def check_args(self, args):
        def wrong_agent(cause): print(f"This agent doesnt have {cause}! Use class AgentCurious"); os._exit(0) 

        if args.has_curiosity:
            wrong_agent('curiosity')            
        
        if args.has_images:
            wrong_agent('state as image support')

    # OVERRIDABLE
    def build_dqn_model(self):
        if self.args.has_curiosity:
            in_features = self.args.encoder_last_layer_out
        else:
            in_features = self.n_states

        return torch.nn.Sequential(
            nn.Linear(in_features=in_features, out_features=self.args.dqn_1_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_1_layer_out, out_features=self.args.dqn_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_2_layer_out, out_features=self.n_actions),
        )


    def end_step(self, reward, next_state, is_done):
        self.e_reward += reward
        self.update_target()

        # Pre populate memory before replay
        if self.memory.get_entries() > self.args.batch_size:
            self.replay(is_done)

        self.total_steps += 1
        self.current_step += 1


    # OVERRIDABLE
    def after_step(self, act_values, reward, next_state, is_terminal):
        transition = [self.current_state, act_values, reward, next_state, is_terminal]
        self.memory.add(transition)

        self.current_state = next_state


    def play_step(self):
        action, act_values = self.act()
        next_state, reward, is_done, _ = self.env.step(action)
        is_terminal = 0.0 if is_done else 1.0

        if self.args.has_normalized_state:
            next_state = normalize_state(next_state)
    
        self.after_step(act_values, reward, next_state, is_terminal) # OVERRIDE THIS
        self.end_step(reward, next_state, is_done) 
        
        return is_done
       
    # CALL AS SUPER
    def reset_env(self):
        state = self.env.reset()
        self.current_state = state
        self.e_loss_dqn.clear()
        self.e_reward = 0
        return state


    # OVERRIDABLE
    def act(self):
        # Pick random action ( Exploration )
        if random.random() <= self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
            act_vector = np.zeros(self.n_actions,) # 1D vect of size 2
            act_vector[action_idx] = 1.0
            return action_idx, act_vector

        # Exploitation
        state_t = torch.as_tensor(self.current_state, dtype=torch.float32, device=self.args.device)

        act_values = self.dqn_model(state_t)  # Predict action based on state
        act_values = to_numpy(act_values)

        action_idx = np.argmax(act_values)
        act_vector = np.zeros(self.n_actions)
        act_vector[action_idx] = 1.0

        return action_idx, act_vector

   
    def train_dqn_model(self, state_t, recorded_action, reward, next_state_t, done, importance_sampling_weight, idxs):
        next_state_Q_val = self.dqn_model(next_state_t)
        next_state_Q_val = to_numpy(next_state_Q_val)

        if self.args.has_ddqn:
            next_state_Q_max_idx = np.argmax(next_state_Q_val, axis=1)
            next_state_target_val = self.target_model(next_state_t)
            next_state_target_val = to_numpy(next_state_target_val)
            next_state_Q_max = next_state_target_val[np.arange(len(next_state_target_val)), next_state_Q_max_idx]
        else:
            next_state_Q_max = np.amax(next_state_Q_val, axis=1)

        # If the game has ended done=0, gets multiplied and extrinsic reward is just itself given this state
        # R(s, a) + gamma * max(Q'(s', a')
        Q_next = np.array(reward + done * self.args.gamma * next_state_Q_max, dtype=np.float)
        Q_next = torch.FloatTensor(Q_next).to(self.args.device)

        Q_cur = self.dqn_model(state_t)
        Q_cur = Q_cur * torch.FloatTensor(recorded_action).to(self.args.device) # gets rid of zeros
        Q_cur = torch.sum(Q_cur, dim=1)   # sum one vect, leaving just qmax

        loss_dqn = self.dqn_model_loss_fn(Q_cur, Q_next) # y_prim, y      LOOK OUT FOR REDUCE PARAMETER!

        if importance_sampling_weight is not None:
            loss_dqn = (torch.FloatTensor(importance_sampling_weight).to(self.args.device) * loss_dqn)

        td_errors = np.abs(to_numpy(Q_next) - to_numpy(Q_cur))

        # PER
        if self.args.has_prioritized:
            self.update_priority(td_errors, idxs)

        return loss_dqn


    def update_target(self):
        self.target_model.load_state_dict(self.dqn_model.state_dict())

    def update_priority(self, td_errors, idxs):
        for i in range(self.args.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_errors[i]) 
        
    def update_target(self):
        if self.args.has_ddqn:
            if self.total_steps % self.args.target_update == 0:
                self.target_model.load_state_dict(self.dqn_model.state_dict())

    # OVERRIDABLE
    def print_debug(self, i_episode, exec_time):
        if self.args.debug:
            dqn_loss = self.loss_dqn[-1]
            ers = self.ers[-1]
            info = f"i_episode: {i_episode}   |   epsilon: {self.epsilon:.4f}   |    dqn:  {dqn_loss:.4f}   |   ers:  {ers:.2f}   |   time: {exec_time:.4f}"
                
            return info

    # CALL AS SUPER
    def terminal_episode(self):
        dqn_avg = average(self.e_loss_dqn)
        self.loss_dqn.append(dqn_avg)
        self.ers.append(self.e_reward)

    # CALL AS SUPER
    def remember_episode(self, loss_dqn):
        self.e_loss_dqn.append(float(loss_dqn)) 

    def backprop(self, loss, is_terminal):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Epsilon decay
        if self.epsilon > self.args.epsilon_floor:
            self.epsilon -= self.args.epsilon_decay
        
        if is_terminal:
            self.terminal_episode()


    def replay(self, is_terminal):     
        minibatch, idxs, importance_sampling_weight = self.memory.get_batch(self.args.batch_size)

        state = np.stack(minibatch[:, 0])
        recorded_action = np.stack(minibatch[:, 1])
        reward = np.stack(minibatch[:, 2])
        next_state = np.stack(minibatch[:, 3])
        done = np.stack(minibatch[:, 4])

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.args.device)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.args.device)
 
        # DQN
        loss_dqn = self.train_dqn_model(state_t, recorded_action, reward, next_state_t, done, importance_sampling_weight, idxs)
        loss_dqn = loss_dqn.mean()

        self.remember_episode(loss_dqn)
        self.backprop(loss_dqn, is_terminal)

            
