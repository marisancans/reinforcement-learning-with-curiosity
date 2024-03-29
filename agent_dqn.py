import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.torch_utils import init_parameters
from modules.torch_utils import to_numpy
from modules.opencv_utils import debug_encoded_states

import random, gym, os, cv2, time
from gym import envs
import numpy as np
from collections import deque
from sum_tree import SumTree, Memory

from collections import deque
from model_builder import FeatureExtractor, ModelBuilder

def average(x):
    if not len(x):
        return 0

    return sum(x) / len(x)


class AgentDQN(nn.Module):
    def __init__(self, args, name):
        super().__init__()

        # --------- AGENT ---------------
        self.name = name
        self.check_args(args)
        self.args = args

        # --------- ENVIROMENT ----------
        self.env = gym.make(self.args.env_name)
        self.current_features_sequence = None # Gets set in self.reset_env(). This holds encoded sequence if enabled

        # --------- ENV STATE --------------- 
        self.n_states = self.env.observation_space.shape[0]
        self.states_sequence = deque(maxlen=self.args.n_sequence)

        self.state_max_val = self.env.observation_space.low.min()
        self.state_min_val = self.env.observation_space.high.max()
        self.n_actions = self.env.action_space.n
        self.epsilon = 1.0

        # --------- MODELS --------------
        if self.args.encoder_type != 'nothing':
            self.feature_extractor = FeatureExtractor(self.args, self.n_states)
        
        builder = ModelBuilder(self.args, self.n_states, self.n_actions)

        if self.args.encoder_type == 'conv':
            builder.encoder_output_size = self.feature_extractor.encoder_output_size

        self.dqn_model = builder.build_dqn_model()
        self.target_model = builder.build_dqn_model()

        if self.args.is_curiosity:
            self.inverse_model = builder.build_inverse_model()
            self.forward_model = builder.build_forward_model()

        # --------   OPTIMIZER AND LOSS  ----
        if self.args.is_curiosity:
            params = list(self.inverse_model.parameters()) + list(self.feature_extractor.encoder.parameters()) + list(self.forward_model.parameters()) + list(self.dqn_model.parameters())
        else:
            params = self.dqn_model.parameters()

        self.optimizer = torch.optim.Adam(params=params, lr = self.args.learning_rate)

        self.dqn_model_loss_fn = nn.MSELoss()
        self.inverse_model_loss_fn = nn.MSELoss()

        # --------- INTERNAL STATE -------
        self.current_episode = 0
        self.total_steps = 0 # in all episodes combined
        self.current_step = 0 # only in current episode
        self.memory = Memory(capacity=self.args.memory_size, is_per=self.args.is_prioritized)

        # ----- TRAINING BUFFER --------
        self.loss_dqn = []
        self.ers = []

        if self.args.is_curiosity:
            self.loss_inverse = []
            self.cos_distance = []
            self.loss_combined = []
        

        # ----- EPISODE BUFFER  --------
        self.e_loss_dqn = []
        self.e_reward = []

        if self.args.is_curiosity:
            self.e_loss_inverse = []
            self.e_cos_distance = []
            self.e_loss_combined = []

        self.update_target()



    # ======    ARG CHECKING =============
    def check_args(self, args):
        if args.batch_size < 4:
            logger('Batch size too small!')
            os._exit(0) 

        if args.is_curiosity:
            if args.curiosity_beta == -1 or args.curiosity_lambda == -1:
                print("Curiosity enabled but lambda or beta value hasnt been set!")
                os._exit(1)

            if args.encoder_type == 'nothing':
                print("Encoder type cant be 'nothing' if curiosity enabled, change the type")
                os._exit(1)

        if args.debug_activations and len(args.debug_activations[0].split()) != 3:
            print('debug_activations len(args) != 3, check help for formatting')
            os._exit(0)

    def normalize_state(self, x):
        d = 2.*(x - self.state_min_val/self.state_max_val) - 1
        return d

    # =======     IMAGE PROCESSING    ====
    def preproprocess_frame(self, frame):
        if self.args.is_grayscale:
            frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
        
        if self.args.image_crop:
            x1, y1, x2, y2 = self.args.image_crop
            frame = frame[y1:y2, x1:x2] 

        s = self.args.image_scale
        if s != 1:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) # can test other interpolation types
        
        # (H, W, C) -> (C, H, W)
        frame = np.moveaxis(frame, -1, 0)

        return frame         

    # =======     SEQUENCE    =============
    def get_features(self):
         # Because when game starts we have just 1 frame and 1 batch size
        sequence_t = torch.stack(list(self.states_sequence))
        seq_lengths = torch.FloatTensor([[len(self.states_sequence)]]) # (batch, frames)
        sequence_t = torch.unsqueeze(sequence_t, 0) # Add batch dim
        sequence_t = self.feature_extractor.extract_features(sequence_t, seq_lengths)
        sequence_t = torch.squeeze(sequence_t, 0) # Remove batch dim
        return sequence_t

    def get_next_sequence(self, next_state_t):        
        self.states_sequence.append(next_state_t)

        # DEBUGGING
        if self.args.debug_images:
           self.debug_sequence()

        
        features = self.get_features()
        return features

    # =====    GAME LOGIC    =============
    def reset_env(self):
        state = self.env.reset()

        if self.args.encoder_type == 'conv':
            state = self.preproprocess_frame(state)

        state_t = torch.FloatTensor(state).to(self.args.device)

        if self.args.is_curiosity: 
            state_t = self.get_next_sequence(state_t)
        
        self.current_features_sequence = state_t

        self.e_loss_dqn.clear()
        self.e_reward .clear()

        if self.args.encoder_type != 'nothing':
            self.e_loss_inverse.clear()
            self.e_loss_combined.clear()
            self.e_cos_distance.clear()

        return state

    def act(self):
        # Pick random action ( Exploration )
        if random.random() <= self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            act_values = self.dqn_model(self.current_features_sequence) 
            _, action_idx = act_values.max(dim=0)
            action_idx = to_numpy(action_idx)

        act_vector = torch.zeros(self.n_actions)
        act_vector[action_idx] = 1.0

        return action_idx, act_vector

    def play_step(self):
        action, act_values_t = self.act()
        next_state, reward, is_terminal, _ = self.env.step(action)

        if self.args.is_normalized_state:
            next_state = self.normalize_state(next_state)

        self.after_step(act_values_t, reward, next_state, is_terminal)
        self.end_step(reward)

        if is_terminal:
            self.terminal_episode()
        
        return is_terminal

    def after_step(self, act_values_t, reward, next_state, is_terminal):
        if self.args.encoder_type == 'conv':
            next_state = self.preproprocess_frame(next_state)

        next_state_t = torch.FloatTensor(next_state).to(self.args.device)

        if self.args.is_curiosity:
            next_state_t = self.get_next_sequence(next_state_t)

        reward_t = torch.FloatTensor([reward]).to(self.args.device)

        t = 0.0 if is_terminal else 1.0
        t = torch.FloatTensor([t]).to(self.args.device)
        transition = [self.current_features_sequence, act_values_t, reward_t, next_state_t, t]
        self.memory.add(transition)

        self.current_features_sequence = next_state_t

    def end_step(self, reward):
        self.e_reward.append(reward)
        self.update_target()

        # Pre populate memory before replay
        if self.memory.get_entries() > self.args.batch_size:
            self.replay()

        self.total_steps += 1
        self.current_step += 1
    
    def replay(self):     
        minibatch, idxs, importance_sampling_weight = self.memory.get_batch(self.args.batch_size)

        state_t = torch.stack([x[0] for x in minibatch])
        recorded_action_t = torch.stack([x[1] for x in minibatch])
        reward_t = torch.stack([x[2] for x in minibatch])
        next_state_t = torch.stack([x[3] for x in minibatch])
        done_t = torch.stack([x[4] for x in minibatch])

        # CURIOSITY LOSS
        if self.args.is_curiosity:
            loss_inv, loss_cos = self.get_inverse_and_forward_loss(state_t, next_state_t, recorded_action_t)  
            reward_t += torch.unsqueeze(loss_cos.detach(), dim=1) * self.args.curiosity_scale
        
        # DQN LOSS
        loss_dqn = self.train_dqn_model(state_t, recorded_action_t, reward_t, next_state_t, done_t, importance_sampling_weight, idxs)

        # LOSS
        if self.args.is_curiosity:
            loss = loss_inv*(1-self.args.curiosity_beta)+self.args.curiosity_beta*loss_cos+self.args.curiosity_lambda*loss_dqn
        else:
            loss = loss_dqn
        
        loss = loss.mean()

        self.remember_episode(loss_dqn)

        if self.args.is_curiosity:
            self.remember_episode_curious(loss_cos, loss_inv, loss)

        self.backprop(loss)

    def terminal_episode(self):
        dqn_avg = average(self.e_loss_dqn)
        self.loss_dqn.append(dqn_avg)
        self.ers.append(sum(self.e_reward))
        self.current_episode += 1

        if self.args.is_curiosity:
            inv_avg = average(self.e_loss_inverse)
            cos_avg = average(self.e_cos_distance)
            com_avg = average(self.e_loss_combined)
            self.loss_inverse.append(inv_avg)
            self.cos_distance.append(cos_avg)
            self.loss_combined.append(com_avg)

    def remember_episode(self, loss_dqn):
        loss_dqn = loss_dqn.detach().mean()
        self.e_loss_dqn.append(float(loss_dqn)) 
    
    def remember_episode_curious(self, loss_cos, loss_inv, loss_com):
        self.e_loss_inverse.append(float(loss_inv))
        self.e_loss_combined.append(float(loss_com))

        loss_cos_avg = average(to_numpy(loss_cos))
        self.e_cos_distance.append(float(loss_cos_avg))
            

    def backprop(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.epsilon > self.args.epsilon_floor:
            self.epsilon -= self.args.epsilon_decay
   
    # ====     AGENT INTERNAL STATE    =====
    def update_target(self):
        self.target_model.load_state_dict(self.dqn_model.state_dict())

    def update_priority(self, td_errors, idxs):
        for i in range(self.args.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_errors[i]) 
        
    def update_target(self):
        if self.args.is_ddqn:
            if self.total_steps % self.args.target_update == 0:
                self.target_model.load_state_dict(self.dqn_model.state_dict())
        
    # ===     DEBUGGING   =====
    def print_debug(self, i_episode, exec_time):
        if self.args.debug:
            dqn_loss = self.loss_dqn[-1] if self.loss_dqn else 0
            ers = sum(self.e_reward)
            info = f"i_episode: {i_episode} | epsilon: {self.epsilon:.4f} |  dqn:  {dqn_loss:.4f} | ers:  {ers:.2f} | time: {exec_time:.2f}"
                
            if self.args.is_curiosity:
                curious_info = f"   |   n_steps: {self.total_steps}   |   mem: {self.memory.get_entries()}   |   com: {self.loss_combined[-1]:.4f}    |    inv: {self.loss_inverse[-1]:.4f}   |   cos: {self.cos_distance[-1]:.4f}"
                info += curious_info

            return info
    
    def get_results(self):
        d = {}
        d['episode'] = self.current_episode
        d['e_score'] = sum(self.e_reward)
        d['e_score_min'] = min(self.e_reward)
        d['e_score_max'] = max(self.e_reward)
        d['score_avg'] = average(self.ers)
        d['score_best'] = max(self.ers)
        d['loss'] = average(self.e_loss_dqn)

        if self.args.is_curiosity:
            d['loss_inverse'] = average(self.loss_inverse)
            d['loss_forward'] = average(self.loss_inverse)
            d['cosine_distance'] = average(self.cos_distance)
        
        return d

    # ================       MODEL TARAINING ====================== 

    def train_dqn_model(self, state_t, recorded_action_t, reward_t, next_state_t, done_t, importance_sampling_weight, idxs):
        next_state_Q_val = self.dqn_model(next_state_t)

        if self.args.is_ddqn:
            _, next_state_Q_max_idx = next_state_Q_val.max(dim=1)
            next_state_Q_max_idx = torch.unsqueeze(next_state_Q_max_idx, dim=1)
            next_state_target_val = self.target_model(next_state_t)
            next_state_Q_max_t = torch.gather(next_state_target_val, dim=1, index=next_state_Q_max_idx)
        else:
            next_state_Q_max_t, _ = next_state_Q_val.max(dim=1, keepdim=True)

        # If the game has ended done=0, gets multiplied and extrinsic reward is just itself given this state
        # R(s, a) + gamma * max(Q'(s', a')
        Q_next = reward_t + done_t * self.args.gamma * next_state_Q_max_t

        Q_cur = self.dqn_model(state_t)
        Q_cur = Q_cur * recorded_action_t 
        Q_cur = torch.sum(Q_cur, dim=1) # gets rid of zeros

        loss_dqn = self.dqn_model_loss_fn(Q_cur, Q_next) # y_prim, y 

        if importance_sampling_weight is not None:
            loss_dqn = (torch.FloatTensor(importance_sampling_weight).to(self.args.device) * loss_dqn)


        # PER
        if self.args.is_prioritized:
            td_errors = torch.abs(torch.squeeze(Q_next, dim=1) - Q_cur)
            self.update_priority(to_numpy(td_errors), idxs)

        return loss_dqn


    def get_inverse_and_forward_loss(self, state_t, next_state_t, recorded_action_t):
        # --------------- INVERSE MODEL -----------------------
        trans = torch.cat((state_t, next_state_t), dim=1)
        pred_action = self.inverse_model(trans)
        loss_inverse = self.inverse_model_loss_fn(pred_action, recorded_action_t)

        # --------------- FORWARD MODEL / CURIOSITY -------------------------
        cat_t = torch.cat((state_t, recorded_action_t), dim=1)

        pred_next_state_t = self.forward_model(cat_t)
        loss_cos = F.cosine_similarity(pred_next_state_t, next_state_t, dim=1)  
        loss_cos = 1.0 - loss_cos

        # DEBUGGING
        if self.args.debug_features:
            debug_encoded_states(pred_next_state, next_state_t)
  
        return loss_inverse, loss_cos