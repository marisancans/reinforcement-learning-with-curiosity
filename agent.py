import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import random, gym, os, cv2, time
from collections import deque
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
        self.current_state = None # Gets set in self.reset_env()

        # --------- STATE --------------- 
        # if image, the state is scaled and cropped
        if self.args.has_images:
            h, w, c = self.env.observation_space.shape
            self.n_states = h * w * self.args.image_scale * self.args.n_sequence_stack# TODO implement scaling and cropping
            self.states_sequence = deque(maxlen=self.args.n_sequence_stack)
        else:
            self.n_states = self.env.observation_space.shape[0]
        
        self.n_actions = self.env.action_space.n

        # MODELS
        self.dqn_model = self.build_dqn_model().to(self.args.device)
        self.target_model = self.build_dqn_model().to(self.args.device)
        self.dqn_model_loss_fn = nn.MSELoss()

        self.current_episode = 0
        self.total_steps = 0 # in all episodes combined
        self.current_step = 0 # only in current episode
        self.update_target()

        self.epsilon = 1.0

        if self.args.has_images:
            self.encoder_model = models.densenet161(pretrained=True)
            # remove last layer
            self.encoder_model = nn.Sequential(*list(self.encoder_model.children())[:-1])
        else:
            self.encoder_model = EncoderModule(self.args, self.n_states).to(self.args.device)

        if self.args.has_curiosity:
            self.inverse_model = self.build_inverse_model().to(self.args.device)
            self.forward_model = self.build_forward_model().to(self.args.device)

            params = list(self.inverse_model.parameters()) + list(self.encoder_model.parameters()) + list(self.forward_model.parameters()) + list(self.dqn_model.parameters())

            self.optimizer = torch.optim.Adam(params=params, lr=self.args.learning_rate)
            self.inverse_model_loss_fn = nn.MSELoss()

            self.loss_inverse = [0]
            self.cos_distance = [0]
            self.loss_combined = [0]

            self.e_loss_inverse = []
            self.e_cos_distance = []
            self.e_loss_combined = []
        else:
            self.optimizer = torch.optim.Adam(params=self.dqn_model.parameters(), lr = self.args.learning_rate)

        self.memory = Memory(capacity=self.args.memory_size)
        self.loss_dqn = [0]
        self.ers = [0]
        
        # ----- EPISODE BUFFER  --------
        self.e_loss_dqn = []
        self.e_reward = 0

    def check_args(self, args):
        if args.has_curiosity:
            if args.curiosity_beta == -1 or args.curiosity_lambda == -1:
                print("Curiosity enabled but lambda or beta value hasnt been set!")
                os._exit(1)

        if args.has_normalized_state:
            if (args.state_min_val > 0 and args.state_max_val < 0) or (args.state_max_val > 0 and args.state_min_val < 0):
                print('Both state_min_val and state_max_val has to be set if manual values are enabled!') 
                os._exit(1)

        if args.has_images:
            if args.dqn_1_layer_out <= 64 or args.dqn_2_layer_out <= 32:
                print('WARNING has_images enabled, but dqn_1_layer or dqn_2_layer out isnt changed, is this a mistake? Image feature vectors usually need bigger layers')
    
            if not args.n_sequence_stack or args.n_sequence_stack <= 1:
                print('n_sequence_stack <= 1 or not passed in arguments')
                os._exit(1)

    def build_dqn_model(self):
        return torch.nn.Sequential(
            nn.Linear(in_features=self.n_states, out_features=self.args.dqn_1_layer_out),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.args.dqn_1_layer_out, out_features=self.args.dqn_2_layer_out),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.args.dqn_2_layer_out, out_features=self.n_actions),
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

    def preproprocess_frame(self, frame):
        # RGB 3 channels to grayscale 1 channel
        return np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])

    def normalize_rows(self, x):
        for row_idx in range(x.shape[0]):
            v = x[row_idx, :]
            row_min = v.min() if self.args.state_min_val == -1 else self.args.state_min_val 
            row_max = v.max() if self.args.state_max_val == -1 else self.args.state_max_val
   
            x[row_idx, :] = (v - row_min) / (row_max - row_min)
        return x

    def get_next_sequence(self, next_state, is_terminal):
        processed = self.preproprocess_frame(next_state)
        if is_terminal:
            blank = np.zeros(shape=(processed.shape[0], processed.shape[1]))
            self.states_sequence.append(blank) # height width
        else:
            self.states_sequence.append(processed)
        
        return np.stack(self.states_sequence)


    def print_debug(self, i_episode, exec_time):
        if self.args.debug:
            dqn_loss = self.loss_dqn[-1]
            ers = self.ers[-1]

            if self.args.debug:
                info = "i_episode: {}   |   epsilon: {:.2f}   |    dqn:  {:.2f}   |   ers:  {:.2f}   |   time: {:.2f}".format(i_episode, self.epsilon, dqn_loss, ers, exec_time)
                
                if self.args.has_curiosity:
                    loss_combined = self.loss_combined[-1]
                    loss_inverse = self.loss_inverse[-1] 
                    cos_distance = self.cos_distance[-1] 

                    info += "   |   com: {:.2f}    |    inv: {:.2f}   |   cos: {:.2f}".format(loss_combined, loss_inverse, cos_distance)

                print(info)

    def play_step(self):
        action, act_values = self.act()
        next_state, reward, is_done, _ = self.env.step(action)
        done = 0.0 if is_done else 1.0

        if self.args.has_images:
            next_state = self.get_next_sequence(next_state, is_done)

        transition = [self.current_state, act_values, reward, next_state, done]
        
        self.memory.add(error=10000, transition=transition) # because its initially unknown and has to be high priority
        self.e_reward += reward
        self.current_state = next_state

        self.update_target()

        # Pre populate memory before replay
        if self.memory.tree.n_entries > self.args.batch_size:
            self.replay(is_done)
        
        self.total_steps += 1
        self.current_step += 1
        return is_done

    def init_current_state(self, state):
        processed = self.preproprocess_frame(state)
        [self.states_sequence.append(processed) for _ in range(self.args.n_sequence_stack)] # Fill with identical states/frames

        return np.stack(self.states_sequence)

    def reset_env(self):
        state = self.env.reset()
        self.current_state = self.init_current_state(state) if self.args.has_images else state
        self.e_loss_dqn.clear()
        self.e_reward = 0
        self.current_step = 0

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
        state_gpu = torch.tensor(self.current_state).float().to(self.args.device)

        if self.args.has_images:
            state_gpu = state_gpu.view(-1)

        act_values = self.dqn_model(state_gpu)  # Predict action based on state
        act_values = act_values.cpu().detach().numpy()

        action_idx = np.argmax(act_values)
        act_vector = np.zeros(self.n_actions)
        act_vector[action_idx] = 1.0

        return action_idx, act_vector

    def get_inverse_and_forward_loss(self, state, next_state, recorded_action):
        # Row wise sate min-max normalization 
        if self.args.has_normalized_state:
            state = self.normalize_rows(state)
            next_state = self.normalize_rows(next_state)

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

        if self.args.debug_states:
            pred_batch = np.array(pred_next_state.detach().numpy() * 255, dtype = np.uint8)
            target_batch = np.array(encoded_next_state.detach().numpy() * 255, dtype = np.uint8)
            pred_batch = np.stack((pred_batch,)*3, axis=-1)
            target_batch = np.stack((target_batch,)*3, axis=-1)

            delim = np.full((pred_batch.shape[0], 1, 3), (0, 0, 255), dtype=np.uint8)
            pred_batch = np.concatenate((pred_batch, delim), axis=1)
            img = np.concatenate((pred_batch, target_batch), axis=1)
            

            cv2.namedWindow('rgb_img', cv2.WINDOW_NORMAL)
            cv2.imshow('rgb_img', img)
            cv2.waitKey(1)
 
  
        return loss_inverse, loss_cos

    def train_dqn_model(self, state, recorded_action, reward, next_state, done, importance_sampling_weight):
        if self.args.has_normalized_state:
            state = self.normalize_rows(state)
            next_state = self.normalize_rows(next_state)

        next_state_gpu = torch.FloatTensor(np.array(next_state)).to(self.args.device)
        state_gpu = torch.FloatTensor(state).to(self.args.device)
        
        # Flatten to get [batch_size, flattened sequences of images]
        if self.args.has_images:
            next_state_gpu = next_state_gpu.view(next_state_gpu.shape[0], -1)
            state_gpu = state_gpu.view(state_gpu.shape[0], -1)

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

        Q_cur = self.dqn_model(state_gpu)
        Q_cur = Q_cur.to('cpu') * torch.FloatTensor(recorded_action) # gets rid of zeros
        Q_cur = torch.sum(Q_cur, dim=1)   # sum one vect, leaving just qmax

        loss_dqn = self.dqn_model_loss_fn(Q_cur, Q_next) # y_prim, y      LOOK OUT FOR REDUCE PARAMETER!
        loss_dqn = (torch.tensor(importance_sampling_weight).float() * loss_dqn)

        td_errors = np.abs(Q_next.detach().numpy() - Q_cur.detach().numpy())

        return td_errors, loss_dqn


    def update_target(self):
        if self.args.has_ddqn:
            if self.total_steps % self.args.target_update == 0:
                self.target_model.load_state_dict(self.dqn_model.state_dict())

    def update_priority(self, td_errors, idxs):
        # update priority
        for i in range(self.args.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_errors[i]) 


    def replay(self, is_terminal):     
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

        # remember loss results
        self.e_loss_dqn.append(float(loss)) 

        if self.args.has_curiosity:
            loss_cos_avg = sum(loss_cos) / len(loss_cos)
            self.e_loss_inverse.append(float(loss_inv))
            self.e_cos_distance.append(float(loss_cos_avg))
            self.e_loss_combined.append(float(loss))
        
        if is_terminal:
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

            

