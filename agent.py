import torch
import torch.nn as nn
import torch.nn.functional as F

import random, gym
from gym import envs
import numpy as np
from collections import deque
from sum_tree import SumTree


def average(x):
    return sum(x) / len(x)

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
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.PER_b)
        is_weight /= is_weight.max()

        return np.array(batch), np.array(idx_arr), is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


#-------------------- AGENT --------------------------
class Agent(nn.Module):
    def __init__(self, args):
        super().__init__()

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

        self.current_episode = 1
        self.total_steps = 1

        if self.args.has_curiosity:
            self.encoder_model = self.build_encoder_model().to(self.device)
            self.inverse_model = self.build_inverse_model().to(self.device)
            self.forward_model = self.build_forward_model().to(self.device)

            params = list(self.inverse_model.parameters())
            params = params + list(self.encoder_model.parameters())
            params = params + list(self.forward_model.parameters())
            params = params + list(self.dqn_model.parameters())

            self.optimizer = torch.optim.Adam(params=params, lr=self.lr)
            self.inverse_model_loss_fn = nn.MSELoss()

            self.loss_inverse = []
            self.cos_distance = []
            self.loss_combined = []

            self.e_loss_inverse = []
            self.e_cos_distance = []
            self.e_loss_combined = []
        else:
            params = list(self.dqn_model.parameters())
            params = params + list(self.target_model.parameters())
            self.optimizer = torch.optim.Adam(params=params, lr= self.lr)

        self.memory = Memory(100000)
        self.loss_dqn = []
        self.ers = []
        
        # ----- EPISODE BUFFER  --------
        self.e_loss_dqn = []
        self.e_reward = 0

    def check_args(self, args):
        if args.has_curiosity:
            if not args.beta_curiosity or not args.lambda_curiosity:
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

    def build_encoder_model(self):
        s_count = self.n_states
        return torch.nn.Sequential(
            nn.Linear(in_features=s_count, out_features=s_count*2),
            nn.ReLU(),
            nn.Linear(in_features=s_count*2, out_features=s_count*3),
            nn.ReLU(),
            nn.Linear(in_features=s_count*3, out_features=s_count*self.encoded_state_out)
        ) 

    def build_inverse_model(self):
        x = 2 # multiply by 2 because we have 2 concated vectors
        return torch.nn.Sequential(
            nn.Linear(in_features=self.n_states*self.encoded_state_out*x, out_features=self.n_states*self.encoded_state_out*2*x),
            nn.ReLU(),
            nn.Linear(in_features=self.n_states*self.encoded_state_out*2*x, out_features=self.n_states*self.encoded_state_out*3*x),
            nn.ReLU(),
            nn.Linear(in_features=self.n_states*self.encoded_state_out*3*x, out_features=self.n_actions)
        )

    def build_forward_model(self):
        encoded_n = self.n_states*self.encoded_state_out
        return torch.nn.Sequential(
            nn.Linear(in_features=encoded_n + self.n_actions, out_features=encoded_n*2), # input actions are one hot encoded
            nn.ReLU(),
            nn.Linear(in_features=encoded_n*2, out_features=encoded_n*3),
            nn.ReLU(),
            nn.Linear(in_features=encoded_n*3, out_features=encoded_n)
        )

    def play_step(self):
        action, act_values = self.act()
        next_state, reward, is_done, _ = self.env.step(action)
        done = 0.0 if is_done else 1.0
        transition = [self.current_state, act_values, reward, next_state, done]
        
        self.memory.add(10000, transition)
        self.e_reward += reward
       
        self.current_episode += 1
        self.current_state = next_state

        if self.ddqn:
            if self.total_steps % self.update_target_every == 0:
                self.update_target()

        if self.current_episode > 3:
            if self.curiosity:
                dqn, inv, cos, com = self.replay()
                self.e_loss_inverse.append(inv)
                self.e_cos_distance.append(cos)
                self.e_loss_combined.append(com)
            else:
                dqn = self.replay()
            
            self.e_loss_dqn.append(dqn)
        


        if is_done:
            dqn_avg = average(self.e_loss_dqn)
            self.loss_dqn.append(dqn_avg)
            self.ers.append(self.e_reward)

            if self.curiosity:
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
        if self.curiosity:
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
        X = torch.tensor(self.current_state).float().to(self.device)
        act_values = self.dqn_model(X)  # Predict action based on state
        act_values = act_values.cpu().detach().numpy()

        action_idx = np.argmax(act_values)
        act_vector = np.zeros(self.n_actions)
        act_vector[action_idx] = 1.0

        return action_idx, act_vector

    def get_inverse_and_forward_loss(self, state, next_state, recorded_action):
        # State normalization
        if self.args.has_normalized_state:
            state /=  state.sum(axis=1)[:, np.newaxis]
            next_state = next_state.sum(axis=1)[:, np.newaxis]

        # State encoding
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
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
        recorded_action_tensor = torch.FloatTensor(recorded_action).to(self.device)
        cat_action_state = torch.cat((encoded_state, recorded_action_tensor), dim=1)

        pred_next_state = self.forward_model(cat_action_state)
        loss_cos = F.cosine_similarity(pred_next_state, encoded_next_state, dim=1)  
        loss_cos = 1.0 - loss_cos
  
        return loss_inverse, loss_cos

    def train_dqn_model(self, state, recorded_action, reward, next_state, done, is_weight):
        next_state_gpu = torch.FloatTensor(np.array(next_state)).to(self.device)
        next_state_Q_val = self.dqn_model(next_state_gpu)
        next_state_Q_val = next_state_Q_val.cpu().detach().numpy()

        if self.ddqn:
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
        curr_state_Q_val = np.array(reward + done * self.gamma * next_state_Q_max, dtype=np.float)
        curr_state_Q_val = torch.FloatTensor(curr_state_Q_val)

        actions = self.dqn_model(torch.FloatTensor(state).to(self.device))
        actions = actions.to('cpu') * torch.FloatTensor(recorded_action) # gets rid of zeros
        actions = torch.sum(actions, dim=1)   # sum one vect, leaving just qmax

        loss_dqn = self.dqn_model_loss_fn(actions, curr_state_Q_val) # y_prim, y      LOOK OUT FOR REDUCE PARAMETER!
        loss_dqn = (torch.tensor(is_weight).float() * loss_dqn)
        loss_dqn = loss_dqn.mean() 

        return actions, curr_state_Q_val, loss_dqn


        # a = self.model.predict(next_state)[0]
        t = self.target_model.predict(next_state)[0]
        target[0][action] = reward + self.gamma * np.amax(t)
        # target[0][action] = reward + self.gamma * t[np.argmax(a)]

    def update_target(self):
            self.target_model.load_state_dict(self.dqn_model.state_dict())

    def update_priority(self, actions, curr_state_Q_val, idxs):
        # priority error
        errors = np.abs(actions.detach().numpy() - curr_state_Q_val.detach().numpy())
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i]) 


    def replay(self):     
        minibatch, idxs, is_weight = self.memory.uniform_segment_batch(self.batch_size)

        state = np.stack(minibatch[:, 0])
        recorded_action = np.stack(minibatch[:, 1])
        reward = np.stack(minibatch[:, 2])
        next_state = np.stack(minibatch[:, 3])
        done = np.stack(minibatch[:, 4])

        # CURIOSITY
        if self.curiosity:
            loss_inv, loss_cos = self.get_inverse_and_forward_loss(state, next_state, recorded_action)  
            intrinsic_reward = loss_cos * self.curiosity_scale
            reward = reward + intrinsic_reward.to("cpu").detach().numpy()
 
        # DQN MODEL 
        actions, curr_state_Q_val, loss_dqn = self.train_dqn_model(state, recorded_action, reward, next_state, done, is_weight)

        # PER
        self.update_priority(actions, curr_state_Q_val, idxs)

        # LOSS
        if self.curiosity:
            loss = loss_inv*(1-self.beta)+self.beta*loss_cos+self.lamda*loss_dqn
            loss = loss.mean()
        else:
            loss = loss_dqn

        loss_dqn.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Epsilon decay
        if self.epsilon > self.epsilon_floor:
            self.epsilon -= self.epsilon_decay

        if self.curiosity:
            loss_cos = sum(loss_cos) / len(loss_cos)
            return float(loss_dqn), float(loss_inv), float(loss_cos), float(loss)
        else:
            return float(loss_dqn)

