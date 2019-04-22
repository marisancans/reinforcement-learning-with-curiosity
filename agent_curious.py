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
from encoder import EncoderModule, SimpleEncoderModule
from agent_dqn import AgentDQN

class AgentCurious(AgentDQN):
    def __init__(self, args, name):
        super().__init__(args, name)

        # --------- ENV STATE --------------- 
        if self.args.has_images:
            h, w = self.calc_image_dims()
            self.n_states = int(h * w * self.args.n_sequence)
            self.states_sequence = deque(maxlen=self.args.n_sequence)

        # --------- MODELS --------------
        if self.args.has_images:
            self.encoder_model = EncoderModule(self.args, self.n_states).to(self.args.device)
        else:
            self.encoder_model = SimpleEncoderModule(self.args, self.n_states).to(self.args.device)
            self.encoder_model = init_parameters('simple encoder', self.encoder_model)

        self.inverse_model = self.build_inverse_model().to(self.args.device)
        self.inverse_model = init_parameters('inverse', self.inverse_model)

        self.forward_model = self.build_forward_model().to(self.args.device)
        self.forward_model = init_parameters('forward', self.forward_model)

        # --------- INTERNAL STATE -------
        params = list(self.inverse_model.parameters()) + list(self.encoder_model.parameters()) + list(self.forward_model.parameters()) + list(self.dqn_model.parameters())

        self.optimizer = torch.optim.Adam(params=params, lr=self.args.learning_rate)
        self.inverse_model_loss_fn = nn.MSELoss()
        
        # ----- TRAINING BUFFER --------
        self.loss_inverse = [0]
        self.cos_distance = [0]
        self.loss_combined = [0]
        
        # ----- EPISODE BUFFER  --------
        self.e_loss_inverse = []
        self.e_cos_distance = []
        self.e_loss_combined = []

    def check_args(self, args):
        if not args.has_curiosity:
            print("Curiosity has to be enabled, if you want to use plain DQN, use AgentDQN class")
            os._exit(0)

        if args.has_curiosity:
            if args.curiosity_beta == -1 or args.curiosity_lambda == -1:
                print("Curiosity enabled but lambda or beta value hasnt been set!")
                os._exit(1)

        if args.debug_activations and len(args.debug_activations[0].split()) != 3:
            print('debug_activations len(args) != 3, check help for formatting')
            os._exit(0)

    def build_dqn_model(self):
        in_features = self.args.encoder_layer_out if self.args.has_images else self.args.simple_encoder_2_layer_out

        return torch.nn.Sequential(
            nn.Linear(in_features=in_features, out_features=self.args.dqn_1_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_1_layer_out, out_features=self.args.dqn_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_2_layer_out, out_features=self.args.dqn_3_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_3_layer_out, out_features=self.args.dqn_4_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_4_layer_out, out_features=self.n_actions),
        )

    def build_inverse_model(self):
        s = 2 # multiply by 2 because we have 2 concated vectors
        in_features = self.args.encoder_layer_out if self.args.has_images else self.args.simple_encoder_2_layer_out

        return torch.nn.Sequential(
            nn.Linear(in_features=in_features * s, out_features=self.args.inverse_1_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.inverse_1_layer_out, out_features=self.args.inverse_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.inverse_2_layer_out, out_features=self.args.inverse_3_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.inverse_3_layer_out, out_features=self.args.inverse_4_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.inverse_4_layer_out, out_features=self.n_actions)
        )

    def build_forward_model(self):
        encoder_out = self.args.encoder_layer_out if self.args.has_images else self.args.simple_encoder_2_layer_out

        return torch.nn.Sequential(
            nn.Linear(in_features=encoder_out + self.n_actions, out_features=self.args.forward_1_layer_out), # input actions are one hot encoded
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_1_layer_out, out_features=self.args.forward_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_2_layer_out, out_features=self.args.forward_3_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_3_layer_out, out_features=encoder_out)
        )

    # calcualte image dimensions to allow dynamc image resizes
    def calc_image_dims(self):
        h, w, c = self.env.observation_space.shape

        # CROPPING
        if self.args.image_crop:
            x1, y1, x2, y2 = self.args.image_crop
            h = y2 - y1
            w = x2 - x1

        # SCALING
        s = self.args.image_scale
        if s != 1:
            h *= s
            w *= s

        return h, w 

    def normalize_state(self, x):
        x = (x - self.state_min_val) / (self.state_max_val - self.state_min_val)
        return x
    
    def init_current_state(self, state):
        processed = self.preproprocess_frame(state)
        [self.states_sequence.append(processed) for _ in range(self.args.n_sequence)] # Fill with identical states/frames

        return self.encode_sequence().detach()

    def encode_state(self, state):
        state_t = torch.FloatTensor(state).to(self.args.device)
        state_t = torch.unsqueeze(state_t, 0)
        state_t = self.encoder_model(state_t)
        state_t = torch.squeeze(state_t, 0)
        state_t = state_t.detach()

        x = to_numpy(state_t)
       
        if np.isnan(x).any():
            a = 1

        return state_t

    def encode_sequence(self):
        states_stack = np.stack(self.states_sequence)
        states_stack_t = torch.FloatTensor(states_stack).to(self.args.device)
        states_stack_t = torch.unsqueeze(states_stack_t, 0) # Add batch dimension
        encoded_state = self.encoder_model(states_stack_t)
        encoded_state = encoded_state.squeeze() # remove batch dimension
        return encoded_state

    def preproprocess_frame(self, frame):
        # RGB 3 channels to grayscale 1 channel
        frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
        
        if self.args.image_crop:
            x1, y1, x2, y2 = self.args.image_crop
            frame = frame[y1:y2, x1:x2] 

        s = self.args.image_scale
        if s != 1:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) # can test other interpolation types
        
        return frame

    def get_next_sequence(self, next_state, is_terminal):
        processed = self.preproprocess_frame(next_state)
        
        if is_terminal:
            blank = np.zeros(shape=(processed.shape[0], processed.shape[1]))
            self.states_sequence.append(blank) # height width
        else:
            self.states_sequence.append(processed)

        # DEBUGGING
        if self.args.debug_images:
            self.debug_sequence()

        encoded_sequence = self.encode_sequence()
        encoded_sequence = encoded_sequence.detach()
        return encoded_sequence

    def debug_sequence(self):
        features = np.array(self.states_sequence * 255, dtype = np.uint8)
        
        #img = np.stack((img,)*3, axis=-1)
        features = np.concatenate(features, axis=1)
        cv2.namedWindow('sequence', cv2.WINDOW_NORMAL)
        cv2.imshow('sequence', features)
        
        cv2.waitKey(1)
        # 
        #self.env.render()

        key = self.args.debug_activations[0]
        features = self.encoder_model.activations[key]
        features = features.squeeze(0).cpu().detach().numpy()
        features = np.array(features * 255, dtype = np.uint8)

        col_count = 10
        height = features.shape[1]
        width = features.shape[2]
        blank_count = col_count - (features.shape[0] % col_count)
        
        # Fill missing feature maps with zeros
        for i in range(blank_count):
            blank = np.zeros(shape=(1, features.shape[1], features.shape[2]), dtype=np.uint8)
            features = np.concatenate((features, blank))

        # Merge all feature maps into 2D image
        features = np.reshape(features, newshape=(-1, col_count, features.shape[1], features.shape[2]))
        row_count = features.shape[0]
        features = np.concatenate(features, axis=1)
        features = np.concatenate(features, axis=1)

        img = np.stack((features,)*3, axis=-1) # Make RGB
        
        # Make grid
        for c, a, s in [[col_count, 1, width], [row_count, 0, height]]:
            for i in range(1, c):
                pos = (s * i + i) - 1
                img = np.insert(img, pos, values=(229, 0, 225), axis=a) 

        cv2.namedWindow('activations', cv2.WINDOW_NORMAL)
        cv2.imshow('activations', img)
        cv2.waitKey(1)

    def print_debug(self, i_episode, exec_time):
        if self.args.debug:
            dqn_loss = self.loss_dqn[-1]
            ers = self.ers[-1]

            if self.args.debug:
                info = f"i_episode: {i_episode}   |   epsilon: {self.epsilon:.4f}   |    dqn:  {dqn_loss:.4f}   |   ers:  {ers:.2f}   |   time: {exec_time:.4f}"
                
                if self.args.has_curiosity:
                    loss_combined = self.loss_combined[-1]
                    loss_inverse = self.loss_inverse[-1] 
                    cos_distance = self.cos_distance[-1] 

                    info += f"   |   com: {loss_combined:.4f}    |    inv: {loss_inverse:.4f}   |   cos: {cos_distance:.4f}"

                print(info)


    def get_inverse_and_forward_loss(self, state_t, next_state_t, recorded_action):
        # --------------- INVERSE MODEL -----------------------
        # transition from s to s_t+1 concatenated column-wise
        trans = torch.cat((state_t, next_state_t), dim=1)

        pred_action = self.inverse_model(trans)
        
        target_action = torch.FloatTensor(recorded_action).to(self.args.device)
        loss_inverse = self.inverse_model_loss_fn(pred_action, target_action)

        # --------------- FORWARD MODEL / CURIOSITY -------------------------
        recorded_action_tensor = torch.FloatTensor(recorded_action).to(self.args.device)
        cat_action_state = torch.cat((state_t, recorded_action_tensor), dim=1)

        pred_next_state = self.forward_model(cat_action_state)
        loss_cos = F.cosine_similarity(pred_next_state, next_state_t, dim=1)  
        loss_cos = 1.0 - loss_cos

        # DEBUGGING
        if self.args.debug_features:
            pred_batch = np.array(pred_next_state.cpu().detach().numpy() * 255, dtype = np.uint8)
            target_batch = np.array(next_state_t.cpu().detach().numpy() * 255, dtype = np.uint8)
            pred_batch = np.stack((pred_batch,)*3, axis=-1)
            target_batch = np.stack((target_batch,)*3, axis=-1)

            delim = np.full((pred_batch.shape[0], 1, 3), (0, 0, 255), dtype=np.uint8)
            pred_batch = np.concatenate((pred_batch, delim), axis=1)
            img = np.concatenate((pred_batch, target_batch), axis=1)
            

            cv2.namedWindow('features', cv2.WINDOW_NORMAL)
            cv2.imshow('features', img)
            cv2.waitKey(1)
  
        return loss_inverse, loss_cos

    def reset_env(self):
        state = super(AgentCurious, self).reset_env()
        
        self.e_loss_inverse.clear()
        self.e_loss_combined.clear()
        self.e_cos_distance.clear()

        if self.args.has_images:
            self.current_state = self.init_current_state(state)
        else:
            self.current_state = self.encode_state(state)


    def update_target(self):
            self.target_model.load_state_dict(self.dqn_model.state_dict())

    def update_priority(self, td_errors, idxs):
        # update priority
        for i in range(self.args.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_errors[i]) 
        
    def update_target(self):
        if self.args.has_ddqn:
            if self.total_steps % self.args.target_update == 0:
                self.target_model.load_state_dict(self.dqn_model.state_dict())

    def sample_batch(self):
        buffer_size = len(self.memory)
        index = np.random.choice(np.arange(buffer_size),
                                size = self.args.batch_size,
                                replace = False)
        
        return np.array([self.memory[i] for i in index])

    def act(self):
        # Pick random action ( Exploration )
        if random.random() <= self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
            act_vector = np.zeros(self.n_actions,) # 1D vect of size 2
            act_vector[action_idx] = 1.0
            return action_idx, act_vector

        # Exploitation
        state_t = self.current_state

        if self.args.has_images:
            state_t = state_t.view(-1)

        act_values = self.dqn_model(state_t)  # Predict action based on state
        act_values = act_values.cpu().detach().numpy()

        action_idx = np.argmax(act_values)
        act_vector = np.zeros(self.n_actions)
        act_vector[action_idx] = 1.0

        return action_idx, act_vector


    def after_step(self, act_values, reward, next_state, done):
        if self.args.has_images:
            next_state = self.get_next_sequence(next_state, is_done)
        else:
            next_state = self.encode_state(next_state) # TODO - This encodes each state, its slow

        transition = [self.current_state, act_values, reward, next_state, done]
        self.memory.add(transition)

        x = to_numpy(next_state)
        ac = to_numpy(self.current_state)

        if np.isnan(x).any() or np.isnan(ac).any():
            a = 1
        
       

        self.current_state = next_state

    def terminal_episode(self):
        super(AgentCurious, self).terminal_episode()

        inv_avg = average(self.e_loss_inverse)
        cos_avg = average(self.e_cos_distance)
        com_avg = average(self.e_loss_combined)
        self.loss_inverse.append(inv_avg)
        self.cos_distance.append(cos_avg)
        self.loss_combined.append(com_avg)

    def remember_episode(self, loss_dqn, loss_cos, loss_inv, loss):
        loss_dqn_avg = average(loss_dqn)
        super(AgentCurious, self).remember_episode(loss_dqn_avg)

        loss_cos_avg = average(loss_cos)
        self.e_loss_inverse.append(float(loss_inv))
        self.e_cos_distance.append(float(loss_cos_avg))
        self.e_loss_combined.append(float(loss))

    def replay(self, is_terminal):     
        minibatch, idxs, importance_sampling_weight = self.memory.get_batch(self.args.batch_size)

        state_t = torch.stack(tuple(minibatch[:, 0]))
        recorded_action = np.stack(minibatch[:, 1])
        reward = np.stack(minibatch[:, 2])
        next_state_t = torch.stack(tuple(minibatch[:, 3]))
        done = np.stack(minibatch[:, 4])


        # CURIOSITY
        loss_inv, loss_cos = self.get_inverse_and_forward_loss(state_t, next_state_t, recorded_action)  
        intrinsic_reward = loss_cos * self.args.curiosity_scale
        reward = reward + intrinsic_reward.cpu().detach().numpy()
 
        loss_dqn = self.train_dqn_model(state_t, recorded_action, reward, next_state_t, done, importance_sampling_weight, idxs)

        # LOSS
        loss = loss_inv*(1-self.args.curiosity_beta)+self.args.curiosity_beta*loss_cos+self.args.curiosity_lambda*loss_dqn
        loss = loss.mean()

        self.remember_episode(loss_dqn, loss_cos, loss_inv, loss)
        self.backprop(loss, is_terminal)

