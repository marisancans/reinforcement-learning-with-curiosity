import torch
import torch.nn as nn
import torch.nn.functional as F

import random, gym, os
from gym import envs
import numpy as np
from collections import deque
from sum_tree import SumTree


def average(x):
    return sum(x) / len(x)

def normalize_rows(x):
    for row_idx in range(x.shape[0]):
        v = x[row_idx, :]   
        x[row_idx, :] = (v - v.min()) / (v.max() - v.min())
    return x

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.PER_e) ** self.PER_a

    def add(self, error, transition):
        p = self._getPriority(error)
        self.tree.add(p, transition) 

    def uniform_segment_batch(self, n):
        batch = []
        idx_arr = []
        priority_arr = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idx_arr.append(idx)
            priority_arr.append(p)

        sampling_probabilities = priority_arr / self.tree.total()
        importance_sampling_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.PER_b)
        importance_sampling_weight /= importance_sampling_weight.max()

        return np.array(batch), np.array(idx_arr), importance_sampling_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

class EncoderModule(nn.Module):
    def __init__(self, args, n_states):
        super(EncoderModule, self).__init__()

        self.seq = torch.nn.Sequential(
            nn.Linear(in_features=n_states, out_features=args.encoder_1_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=args.encoder_1_layer_out, out_features=args.encoder_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=args.encoder_2_layer_out, out_features=args.encoder_3_layer_out),
            nn.Tanh()
        ) 

    def forward(self, x):
        embedding = self.seq(x)
        
        # L2 normalization
        norm = torch.norm(embedding.detach(), p=2, dim=1, keepdim=True)
        output_norm = embedding / norm
        return output_norm


#-------------------- AGENT --------------------------
class Agent(nn.Module):
    def __init__(self, args, name):
        super().__init__()

        self.name = name
        self.check_args(args)
        self.args = args

        # --------- ENVIROMENT ----------
        self.env = gym.make(self.args.env_name)
        self.current_state = self.env.reset()
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        # MODELS
        self.dqn_model = self.build_dqn_model().to(self.args.device)
        self.target_model = self.build_dqn_model().to(self.args.device)
        self.update_target()
        self.dqn_model_loss_fn = nn.MSELoss()

        self.current_episode = 0
        self.total_steps = 0
        self.epsilon = 1.0

        if self.args.has_curiosity:
            self.encoder_model = EncoderModule(self.args, self.n_states).to(self.args.device)
            self.inverse_model = self.build_inverse_model().to(self.args.device)
            self.forward_model = self.build_forward_model().to(self.args.device)

            # \ is not a division operator!
            params = list(self.inverse_model.parameters()) + list(self.encoder_model.parameters()) + list(self.forward_model.parameters()) + list(self.dqn_model.parameters())

            self.optimizer = torch.optim.Adam(params=params, lr=self.args.learning_rate)
            self.inverse_model_loss_fn = nn.MSELoss()

            self.loss_inverse = []
            self.cos_distance = []
            self.loss_combined = []

            self.e_loss_inverse = []
            self.e_cos_distance = []
            self.e_loss_combined = []
        else:
            self.optimizer = torch.optim.Adam(params=self.dqn_model.parameters(), lr = self.args.learning_rate)

        self.memory = Memory(capacity=self.args.memory_size)
        self.loss_dqn = []
        self.ers = [0]
        
        # ----- EPISODE BUFFER  --------
        self.e_loss_dqn = []
        self.e_reward = 0

    def check_args(self, args):
        if args.has_curiosity:
            if args.curiosity_beta == -1 or args.curiosity_lambda == -1:
                print("Curiosity enabled but lambda or beta value hasnt been set!")
                os._exit(1)
    
    def build_dqn_model(self):
        return torch.nn.Sequential(
            nn.Linear(in_features=self.n_states, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=self.n_actions),
        )

    def build_inverse_model(self):
        # multiply by 2 because we have 2 concated vectors
        return torch.nn.Sequential(
            nn.Linear(in_features=self.args.encoder_3_layer_out * 2, out_features=self.args.inverse_1_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.inverse_1_layer_out, out_features=self.args.inverse_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.inverse_2_layer_out, out_features=self.n_actions)
        )

    def build_forward_model(self):
        return torch.nn.Sequential(
            nn.Linear(in_features=self.args.encoder_3_layer_out + self.n_actions, out_features=self.args.forward_1_layer_out), # input actions are one hot encoded
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_1_layer_out, out_features=self.args.forward_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_2_layer_out, out_features=self.args.encoder_3_layer_out)
        )

    def play_step(self):
        action, act_values = self.act()
        next_state, reward, is_done, _ = self.env.step(action)
        done = 0.0 if is_done else 1.0
        transition = [self.current_state, act_values, reward, next_state, done]
        
        self.memory.add(error=10000, transition=transition) # because its initially unknown and has to be high priority
        self.e_reward += reward
        self.current_state = next_state

        if self.args.has_ddqn:
            if self.total_steps % self.args.target_update == 0:
                self.update_target()

        has_collected = self.memory.tree.n_entries > self.args.batch_size
        
        if has_collected:
            if self.args.has_curiosity:
                dqn, inv, cos, com = self.replay()
                self.e_loss_inverse.append(inv)
                self.e_cos_distance.append(cos)
                self.e_loss_combined.append(com)
            else:
                dqn = self.replay()
            
            self.e_loss_dqn.append(dqn)
        
        if is_done and has_collected:
            dqn_avg = average(self.e_loss_dqn)
            self.loss_dqn.append(dqn_avg)
            self.ers.append(self.e_reward)

            if self.args.has_curiosity:
                inv_avg = average(self.e_loss_inverse)
                cos_avg = average(self.e_cos_distance)
                com_avg = average(self.e_loss_combined)
                self.loss_inverse.append(inv_avg)
                self.cos_distance.append(cos_avg)
                self.loss_combined.append(com_avg)

        self.total_steps += 1
        return is_done


    def reset_env(self):
        self.current_state = self.env.reset()
        self.e_loss_dqn.clear()
        self.e_reward = 0
        if self.args.has_curiosity:
            self.e_loss_inverse.clear()
            self.e_loss_combined.clear()
            self.e_cos_distance.clear()


    def forward(self, input):
        return self.layers.forward(input)

    def remember(self, state, act_values, reward, next_state, done, priority):
        #                    s       a         r t+1     s t+1      1/0   
        self.memory.add(priority, [state, act_values, reward, next_state, done])


    # Returns np.zeros 1D vector where 1.0 is the selected action and action idx
    # saving it as a vector helps to do array wise calculations in Bellman equation
    def act(self):
        # Pick random action ( Exploration )
        if random.random() <= self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
            act_vector = np.zeros(self.n_actions,) # 1D vect of size 2
            act_vector[action_idx] = 1.0
            return action_idx, act_vector

        # Exploitation
        X = torch.tensor(self.current_state).float().to(self.args.device)
        act_values = self.dqn_model(X)  # Predict action based on state
        act_values = act_values.cpu().detach().numpy()

        action_idx = np.argmax(act_values)
        act_vector = np.zeros(self.n_actions)
        act_vector[action_idx] = 1.0

        return action_idx, act_vector

    def get_inverse_and_forward_loss(self, state, next_state, recorded_action):
        # Row wise sate min-max normalization 
        if self.args.has_normalized_state:
            state = normalize_rows(state)
            next_state = normalize_rows(next_state)

        # State encoding
        state_tensor = torch.FloatTensor(state).to(self.args.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.args.device)
        encoded_state = self.encoder_model(state_tensor)
        encoded_next_state = self.encoder_model(next_state_tensor)

        # --------------- INVERSE MODEL -----------------------
        # transition from s to s_t+1 concatenated column-wise
        trans = torch.cat((encoded_state, encoded_next_state), dim=1)

        pred_action = self.inverse_model(trans)
        pred_action = pred_action.to('cpu')
        
        target_action = torch.FloatTensor(recorded_action)
        loss_inverse = self.inverse_model_loss_fn(pred_action, target_action)

        # --------------- FORWARD MODEL / CURIOSITY -------------------------
        recorded_action_tensor = torch.FloatTensor(recorded_action).to(self.args.device)
        cat_action_state = torch.cat((encoded_state, recorded_action_tensor), dim=1)

        pred_next_state = self.forward_model(cat_action_state)
        loss_cos = F.cosine_similarity(pred_next_state, encoded_next_state, dim=1)  
        loss_cos = 1.0 - loss_cos
  
        return loss_inverse, loss_cos

    def train_dqn_model(self, state, recorded_action, reward, next_state, done, importance_sampling_weight):
        if self.args.has_normalized_state:
            state = normalize_rows(state)
            next_state = normalize_rows(next_state)

        next_state_gpu = torch.FloatTensor(np.array(next_state)).to(self.args.device)
        next_state_Q_val = self.dqn_model(next_state_gpu)
        next_state_Q_val = next_state_Q_val.cpu().detach().numpy()

        if self.args.has_ddqn:
            next_state_Q_max_idx = np.argmax(next_state_Q_val, axis=1)

            # DDQN
            # https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn
            next_state_target_val = self.target_model(next_state_gpu)
            next_state_target_val = next_state_target_val.cpu().detach().numpy()
            next_state_Q_max = next_state_target_val[np.arange(len(next_state_target_val)), next_state_Q_max_idx]
        else:
            next_state_Q_max = np.amax(next_state_Q_val, axis=1)

        # If the game has ended done=0, gets multiplied and extrinsic reward is just itself given this state
        # R(s, a) + gamma * max(Q'(s', a')
        Q_next = np.array(reward + done * self.args.gamma * next_state_Q_max, dtype=np.float)
        Q_next = torch.FloatTensor(Q_next)

        Q_cur = self.dqn_model(torch.FloatTensor(state).to(self.args.device))
        Q_cur = Q_cur.to('cpu') * torch.FloatTensor(recorded_action) # gets rid of zeros
        Q_cur = torch.sum(Q_cur, dim=1)   # sum one vect, leaving just qmax

        loss_dqn = self.dqn_model_loss_fn(Q_cur, Q_next) # y_prim, y      LOOK OUT FOR REDUCE PARAMETER!
        loss_dqn = (torch.tensor(importance_sampling_weight).float() * loss_dqn)

        td_errors = np.abs(Q_next.detach().numpy() - Q_cur.detach().numpy())

        return td_errors, loss_dqn


    def update_target(self):
            self.target_model.load_state_dict(self.dqn_model.state_dict())

    def update_priority(self, td_errors, idxs):
        # update priority
        for i in range(self.args.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_errors[i]) 


    def replay(self):     
        minibatch, idxs, importance_sampling_weight = self.memory.uniform_segment_batch(self.args.batch_size)

        state = np.stack(minibatch[:, 0])
        recorded_action = np.stack(minibatch[:, 1])
        reward = np.stack(minibatch[:, 2])
        next_state = np.stack(minibatch[:, 3])
        done = np.stack(minibatch[:, 4])

        # CURIOSITY
        if self.args.has_curiosity:
            loss_inv, loss_cos = self.get_inverse_and_forward_loss(state, next_state, recorded_action)  
            intrinsic_reward = loss_cos * self.args.curiosity_scale
            reward = reward + intrinsic_reward.to("cpu").detach().numpy()
 
        # DQN MODEL 
        td_errors, loss_dqn = self.train_dqn_model(state, recorded_action, reward, next_state, done, importance_sampling_weight)

        # PER
        self.update_priority(td_errors, idxs)

        # LOSS
        if self.args.has_curiosity:
            loss = loss_inv*(1-self.args.curiosity_beta)+self.args.curiosity_beta*loss_cos+self.args.curiosity_lambda*loss_dqn
        else:
            loss = loss_dqn

        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Epsilon decay
        if self.epsilon > self.args.epsilon_floor:
            self.epsilon -= self.args.epsilon_decay

        if self.args.has_curiosity:
            loss_cos = sum(loss_cos) / len(loss_cos)
            return float(loss_dqn.detach().mean()), float(loss_inv), float(loss_cos), float(loss)
        else:
            return float(loss)

