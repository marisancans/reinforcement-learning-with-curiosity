import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import random, gym, os, cv2, time, re
from gym import envs
import numpy as np
from collections import deque
from sum_tree import SumTree, Memory

from collections import deque

def average(x):
    return sum(x) / len(x)


class SimpleEncoderModule(nn.Module):
    def __init__(self, args, n_states):
        super(SimpleEncoderModule, self).__init__()

        self.seq = torch.nn.Sequential(
                nn.Linear(in_features=n_states, out_features=args.encoder_1_layer_out),
                nn.ReLU(),
                nn.Linear(in_features=args.encoder_1_layer_out, out_features=args.encoder_2_layer_out),
                nn.ReLU(),
                nn.Linear(in_features=args.encoder_2_layer_out, out_features=args.encoder_out),
                nn.Tanh()
            ) 

    def forward(self):
        embedding = self.seq(x)
        
        # L2 normalization
        norm = torch.norm(embedding.detach(), p=2, dim=1, keepdim=True)
        output_norm = embedding / norm
        return output_norm


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

class EncoderModule(nn.Module):
    def __init__(self, args, n_states):
        super(EncoderModule, self).__init__()
        self.has_images = args.has_images
        self.args = args
        self.n_state = n_states

        self.activations = {}
        self.dense = models.densenet161(pretrained=True).to(self.args.device)
        self.adjust_layers()
      
        
        # Add hook to desired layer. This should be done with recursion
        if args.debug_activations:
            denseblock, denselayer, conv = args.debug_activations[0].split()
            for name, module in self.dense.features.named_children():
                if re.sub('denseblock([0-9]*)$',r'\1', name) == denseblock:
                    for namedb, moduledb in module.named_children():
                        if re.sub('denselayer([0-9]*)$',r'\1', namedb) == denselayer:
                            for namel, modulel in moduledb.named_children():
                                if re.sub('conv([0-9]*)$',r'\1', namel) == conv:
                                    key = args.debug_activations[0]
                                    modulel.register_forward_hook(self.get_activation(key))


    def forward(self, x):
        features = self.dense(x)
               
        return features

    def adjust_layers(self):
        # The very first conv layer of dense net. This has to be adjusted to custom channel count
        conv0_pretrained = list(self.dense.features.children())[0] 

        if conv0_pretrained.in_channels != self.args.n_sequence:
            conv0_pretrained_weights = conv0_pretrained.weight.data
            conv0 = torch.nn.Conv2d(in_channels=self.args.n_sequence, out_channels=96, kernel_size=7, stride=2, padding=3, bias=False) #struct copied from densenet
            conv0_pretrained_in = conv0_pretrained.in_channels

            # If we have 3 channels but 2 are wanted, copy only first 2

            # Else copy all 3, then copy other n wanted channels RANDOMLY from inital pretrained weights
            # Example. If we have weights (a, b, c) size 3, but we want size 5, we will get (a, b, c, [a or b or c], [a or b or c])
            if self.args.n_sequence < conv0_pretrained_in:
                conv0.weight.data = conv0_pretrained_weights[:, 0:self.args.n_sequence, :, :]
            else:
                new_dims = self.args.n_sequence - conv0_pretrained_in
                
                for _ in range(new_dims):
                    rnd_channel = random.randint(0, conv0_pretrained_weights.shape[1] - 1)
                    conv0_pretrained_weights = torch.cat((conv0_pretrained_weights, conv0_pretrained_weights[:, rnd_channel:rnd_channel + 1, :]), dim=1)
                   
            self.dense._modules['features'][0] = conv0

        self.dense._modules['classifier'] = nn.Linear(2208, self.args.encoder_last_layer_out)
        # self.num_features = nn.Sequential(*list(self.dense.children())[0])[-1].num_features # get output feature count from last layer
    

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook


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
        # calcualte image dimensions to allow dynamc image resizes
        if self.args.has_images:
            h, w = self.calc_image_dims()
            
            self.n_states = int(h * w * self.args.n_sequence)# TODO implement scaling and cropping
            self.states_sequence = deque(maxlen=self.args.n_sequence)
        else:
            self.n_states = self.env.observation_space.shape[0]

        self.state_max_val = self.env.observation_space.low.min()
        self.state_min_val = self.env.observation_space.high.max()
        self.n_actions = self.env.action_space.n
        self.epsilon = 1.0

        # MODELS
        self.dqn_model = self.build_dqn_model().to(self.args.device)
        self.target_model = self.build_dqn_model().to(self.args.device)
        self.dqn_model_loss_fn = nn.MSELoss()

        self.current_episode = 0
        self.total_steps = 0 # in all episodes combined
        self.current_step = 0 # only in current episode
        self.update_target()

        if self.args.has_curiosity:
            self.encoder_model = EncoderModule(self.args, self.n_states).to(self.args.device)
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

        if args.has_images:
            if args.dqn_1_layer_out <= 64 or args.dqn_2_layer_out <= 32:
                print('has_images enabled, but dqn_1_layer or dqn_2_layer out isnt changed, is this a mistake? Image feature vectors usually need bigger layers')

        if args.debug_activations and len(args.debug_activations[0].split()) != 3:
            print('debug_activations len(args) != 3')
            os._exit(0)

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
            nn.Linear(in_features=self.args.dqn_2_layer_out, out_features=self.args.dqn_3_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_3_layer_out, out_features=self.args.dqn_4_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_4_layer_out, out_features=self.n_actions),
        )

    def build_inverse_model(self):
        # multiply by 2 because we have 2 concated vectors
        return torch.nn.Sequential(
            nn.Linear(in_features=self.args.encoder_last_layer_out * 2, out_features=self.args.inverse_1_layer_out),
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
        return torch.nn.Sequential(
            nn.Linear(in_features=self.args.encoder_last_layer_out + self.n_actions, out_features=self.args.forward_1_layer_out), # input actions are one hot encoded
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_1_layer_out, out_features=self.args.forward_2_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_2_layer_out, out_features=self.args.forward_3_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_3_layer_out, out_features=self.args.encoder_last_layer_out)
        )

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

        return self.encode_sequence()

    def encode_sequence(self):
        states_stack = np.stack(self.states_sequence)
        states_stack_t = torch.FloatTensor(states_stack).to(self.args.device)
        states_stack_t = torch.unsqueeze(states_stack_t, 0) # Add batch dimension
        encoded_state = self.encoder_model(states_stack_t)
        encoded_state = encoded_state.squeeze() # remove batch dimension
        return encoded_state.cpu().detach().numpy()

    def preproprocess_frame(self, frame):
        # RGB 3 channels to grayscale 1 channel
        frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])

        if self.args.has_normalized_state:
            frame = self.normalize_state(frame)
        
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

    def play_step(self):
        action, act_values = self.act()
        next_state, reward, is_done, _ = self.env.step(action)
        done = 0.0 if is_done else 1.0

        if self.args.has_normalized_state:
            next_state = self.normalize_state(next_state)

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


    def reset_env(self):
        state = self.env.reset()
        self.current_state = self.init_current_state(state) if self.args.has_images else state
        self.e_loss_dqn.clear()
        self.e_reward = 0
        if self.args.has_curiosity:
            self.e_loss_inverse.clear()
            self.e_loss_combined.clear()
            self.e_cos_distance.clear()


    def forward(self, input):
        return self.layers.forward(input)

    def remember(self, state, act_values, reward, next_state, done, priority):
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

    def train_dqn_model(self, state_t, recorded_action, reward, next_state_t, done, importance_sampling_weight):
        next_state_Q_val = self.dqn_model(next_state_t)
        next_state_Q_val = next_state_Q_val.cpu().detach().numpy()

        if self.args.has_ddqn:
            next_state_Q_max_idx = np.argmax(next_state_Q_val, axis=1)

            # DDQN
            # https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn
            next_state_target_val = self.target_model(next_state_t)
            next_state_target_val = next_state_target_val.cpu().detach().numpy()
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
        loss_dqn = (torch.FloatTensor(importance_sampling_weight).to(self.args.device) * loss_dqn)

        td_errors = np.abs(Q_next.cpu().detach().numpy() - Q_cur.cpu().detach().numpy())

        return td_errors, loss_dqn


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


    def replay(self, is_terminal):     
        minibatch, idxs, importance_sampling_weight = self.memory.uniform_segment_batch(self.args.batch_size)

        state = np.stack(minibatch[:, 0])
        recorded_action = np.stack(minibatch[:, 1])
        reward = np.stack(minibatch[:, 2])
        next_state = np.stack(minibatch[:, 3])
        done = np.stack(minibatch[:, 4])

        state_t = torch.FloatTensor(state).to(self.args.device)
        next_state_t = torch.FloatTensor(np.array(next_state)).to(self.args.device)

        # CURIOSITY
        if self.args.has_curiosity:
            loss_inv, loss_cos = self.get_inverse_and_forward_loss(state_t, next_state_t, recorded_action)  
            intrinsic_reward = loss_cos * self.args.curiosity_scale
            reward = reward + intrinsic_reward.cpu().detach().numpy()
 
        # DQN MODEL 
        # Flatten to get [batch_size, flattened sequences of images]
        if self.args.has_images:
            next_state_t = next_state_t.view(next_state_t.shape[0], -1)
            state_t = state_t.view(state_t.shape[0], -1)

        td_errors, loss_dqn = self.train_dqn_model(state_t, recorded_action, reward, next_state_t, done, importance_sampling_weight)

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
