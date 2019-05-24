import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.torch_utils import init_parameters
from modules.torch_utils import to_numpy

import random, gym, os, cv2, time, logging
from gym import envs
import numpy as np
from collections import deque
from memory import Memory

from collections import deque
from models import FeatureEncoder, FeatureDecoder, ModelBuilder, DQN_model

def average(x):
    if not len(x):
        return 0

    return sum(x) / len(x)


class Agent(nn.Module):
    def __init__(self, args, name):
        super().__init__()

        # --------- AGENT ---------------
        self.name = name
        self.check_args(args)
        self.args = args
        logging.info('agent args ok')

        # --------- ENVIROMENT ----------
        self.env = gym.make(self.args.env_name)
        self.current_state = None # Gets set in self.reset_env(). This holds encoded sequence if enabled
        logging.info('agent enviroment ok')

        # --------- ENV STATE --------------- 
        self.n_states = self.env.observation_space.shape[0]
        self.states_sequence = deque(maxlen=self.args.n_sequence)

        self.state_max_val = self.env.observation_space.low
        self.state_min_val = self.env.observation_space.high
        self.n_actions = self.env.action_space.n
        self.epsilon = 1.0
        logging.info('agent state ok')

        # --------- MODELS --------------
        if self.args.encoder_type != 'nothing':
            self.feature_encoder = FeatureEncoder(self.args, self.n_states)
            encoder_out = self.feature_encoder.encoder_output_size
            self.env.reset()
           
            if self.args.encoder_type == 'conv':
                if self.args.render_xvfb:
                    img = self.env.render(mode='rgb_array')
                    original_shape = img.shape 
                else:
                    original_shape = self.env.observation_space.high.shape
            else:
                original_shape = None
            
            self.feature_decoder = FeatureDecoder(args, encoder_out, self.n_states, original_shape=original_shape)
        
        builder = ModelBuilder(self.args, self.n_states, self.n_actions)

        if self.args.encoder_type == 'conv':
            builder.encoder_output_size = encoder_out

        self.dqn_model = builder.build_dqn_model()
        self.target_model = builder.build_dqn_model()

        if self.args.is_curiosity:
            self.inverse_model = builder.build_inverse_model()
            self.forward_model = builder.build_forward_model()
        logging.info('agent models ok')

        # --------   OPTIMIZER AND LOSS  ----
        if self.args.is_curiosity:
            params = list(self.inverse_model.parameters()) + list(self.feature_encoder.encoder.parameters()) + list(self.forward_model.parameters()) + list(self.dqn_model.parameters())
        else:
            params = self.dqn_model.parameters()

        self.optimizer_agent = torch.optim.Adam(params=params, lr = self.args.learning_rate)

        if self.args.encoder_type != 'nothing':
            params_auto_encoder = list(self.feature_encoder.encoder.parameters()) + list(self.feature_decoder.decoder.parameters())
            self.optimizer_autoencoder = torch.optim.Adam(params=params_auto_encoder, lr = self.args.learning_rate)

        logging.info('agent optimizer and params ok')

        # --------- INTERNAL STATE -------
        self.current_episode = 0
        self.total_steps = 0 # in all episodes combined
        self.current_step = 0 # only in current episode
        self.memory = Memory(self.args)
        logging.info('agent memory ok')

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

        if self.args.load_path:
            self.load_models(self.args.load_path)



    # ======    ARG CHECKING =============
    def check_args(self, args):
        if args.batch_size < 4:
            logging.critical('Batch size too small!')
            os._exit(0) 

        if args.is_curiosity:
            if args.curiosity_beta == -1 or args.curiosity_lambda == -1:
                logging.critical("Curiosity enabled but lambda or beta value hasnt been set!")
                os._exit(1)

            if args.encoder_type == 'nothing':
                logging.critical("Encoder type cant be 'nothing' if curiosity enabled, change is_curiosity to false")
                os._exit(1)
            
            if args.encoder_type == 'conv':
                if not torch.cuda.is_available():
                    logging.critical('Cuda not detected')
                    os._exit(0)

                if args.image_crop != 0:
                    if len(args.image_crop) != 4:
                        logging.critical('Image crop has to have 4 coordinates!')
                        os._exit(0)

        if args.debug_features:
            if args.debug_activations and len(args.debug_activations[0].split()) != 3:
                logging.critical('debug_activations len(args) != 3, check help for formatting')
                os._exit(0)
        if args.prioritized_type != 'random':
            if args.per_b_anneal_to == 1:
                logging.critical('prioritized_type is proportional but per_b_anneal_to hasnt been set')
                os._exit(0)

        if args.prioritized_type == 'rank':
            if not args.rank_update:
                logging.critical('prioritized_type is rank, but rank_update hasnt been set')
                os._exit(0)

        if args.memory_size < args.batch_size:
            logging.critical('Memory size has to be larger than batch size')
            os._exit(0)

        if args.load_path:
            if not os.path.exists(args.load_path):
                logging.critical('Load folder not found!')
                os._exit(0)

    def load_models(self, path):
        device = self.args.device

        dqn = path + '/' + 'dqn.pth'
        dqn_target = path + '/' + 'dqn_target.pth'
        optimizer_agent = path + '/' + 'optimizer_agent.pth'

        dqn = torch.load(dqn, map_location=device)
        dqn_target = torch.load(dqn_target, map_location=device)
        optimizer_agent = torch.load(optimizer_agent, map_location=device)
        
        self.dqn_model.load_state_dict(dqn)
        self.target_model.load_state_dict(dqn_target)
        self.optimizer_agent.load_state_dict(optimizer_agent)



        if self.args.is_curiosity:
            forward = path + '/' + 'forward.pth'
            inverse = path + '/' + 'inverse.pth'

            forward = torch.load(forward, map_location=device)
            inverse = torch.load(inverse, map_location=device)

            self.forward_model.load_state_dict(forward)
            self.inverse_model.load_state_dict(inverse)
        
        if self.args.encoder_type != 'nothing':
            encoder = path + '/' + 'encoder.pth'
            decoder = path + '/' + 'decoder.pth'
            optimizer_autoencoder = path + '/' + 'optimizer_autoencoder.pth'

            encoder = torch.load(encoder, map_location=device)
            decoder = torch.load(decoder, map_location=device)
            optimizer_autoencoder = torch.load(optimizer_autoencoder, map_location=device)

            self.feature_encoder.encoder.load_state_dict(encoder)
            self.feature_decoder.decoder.load_state_dict(decoder)
            self.optimizer_autoencoder.load_state_dict(optimizer_autoencoder)
        logging.info('Modules loaded from checkpoint')


    # Normalize -1..1
    def normalize_state(self, x):
        if not np.array_equal(self.state_max_val, self.state_min_val):
            x = (x - self.state_min_val) / (self.state_max_val - self.state_min_val)
        return x


    # =======     IMAGE PROCESSING    ====
    def preproprocess_frame(self, frame):
        if self.args.is_grayscale:
            frame = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])

        if self.args.image_crop != 0:
            x1, y1, x2, y2 = self.args.image_crop
            frame = frame[y1:y2, x1:x2] 

        s = self.args.image_scale

        if s != 1.0:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) # can test other interpolation types
        
        # (H, W, C) -> (C, H, W)
        frame = np.moveaxis(frame, -1, 0)

        return frame         

    # =======     SEQUENCE    =============
    def encode_sequence(self):
         # Because when game starts we have just 1 frame and 1 batch size
        sequence_t = torch.stack(list(self.states_sequence))
        sequence_t = torch.unsqueeze(sequence_t, 0) # Add batch dim
        sequence_t = self.feature_encoder.extract_features(sequence_t)
        sequence_t = torch.squeeze(sequence_t, 0) # Remove batch dim
        return sequence_t

    def encode_sequence_with_next_state(self, next_state_t):
        if self.args.is_grayscale:
            next_state_t = next_state_t.unsqueeze(0)

        self.states_sequence.append(next_state_t)

        features = self.encode_sequence()
        return features

    def decode_sequence(self, truth, h_vector):
        mse = nn.MSELoss()

        if self.args.encoder_type == 'simple':
            pred = self.feature_decoder.decode_simple_features(h_vector)   
        else:
            pred = self.feature_decoder.decode_conv_features(h_vector)
            pred = pred.squeeze(0)
            # debug_auto(pred, truth)  
            
        loss = mse(pred, truth) * self.args.decoder_coeficient
        loss.backward(retain_graph=True)

        #print(float(loss))
        
        self.optimizer_autoencoder.step()
        self.optimizer_autoencoder.zero_grad()

    # =====    GAME LOGIC    =============
    def reset_env(self):
        state = self.env.reset()
        if self.args.render_xvfb:
            state = self.env.render(mode='rgb_array')

        # If is image
        if len(state.shape) == 3:
            state = self.preproprocess_frame(state)

        if self.args.render_xvfb:
            state_t = torch.from_numpy(np.flip(state, axis=0).copy()).float()# Fixes wierd torch error
        else:
            state_t = torch.FloatTensor(state)

        if self.args.encoder_type != 'nothing': 
            state_t = self.encode_sequence_with_next_state(state_t)
            state_t = state_t.detach()
        
        self.current_state = state_t

        self.e_loss_dqn.clear()
        self.e_reward .clear()

        if self.args.is_curiosity:
            self.e_loss_inverse.clear()
            self.e_loss_combined.clear()
            self.e_cos_distance.clear()

        return state

    def act(self):
        # Explore
        if random.random() <= self.epsilon:
            action_idx = random.randint(0, self.n_actions - 1)
        # Exploit
        else:
            self.dqn_model.eval()
            with torch.no_grad():
                act_values = self.dqn_model(self.current_state)
            self.dqn_model.train()
            _, action_idx = act_values.max(dim=0)
            action_idx = to_numpy(action_idx)

        act_vector_t = torch.zeros(self.n_actions).to(self.args.device)
        act_vector_t[action_idx] = 1.0

        return action_idx, act_vector_t

    def play_step(self):
        action, act_vector_t = self.act()
        next_state, reward, is_terminal, _ = self.env.step(action)

        if self.args.render_xvfb:
            next_state = self.env.render(mode='rgb_array')

        if self.args.is_normalized_state:
            next_state = self.normalize_state(next_state)

        self.after_step(act_vector_t, reward, next_state, is_terminal)
        self.end_step(reward)

        if is_terminal:
            self.terminal_episode()
            self.memory.reset_beta()

        # if self.current_episode > 200:
            # self.env.render() 
        
        return is_terminal

    def after_step(self, act_vector_t, reward, next_state, is_terminal):
        if len(next_state.shape) == 3:
            next_state = self.preproprocess_frame(next_state)

        if self.args.render_xvfb:
            next_state_t = torch.from_numpy(np.flip(next_state, axis=0).copy()).float()# Fixes wierd torch error
        else:
            next_state_t = torch.FloatTensor(next_state)
        
        reward_t = torch.FloatTensor([reward]).to(self.args.device)

        if self.args.encoder_type != 'nothing':
            # Auto Encoder training
            target_state_t = next_state_t.to(self.args.device)
            next_state_t = self.encode_sequence_with_next_state(next_state_t)
            self.decode_sequence(target_state_t, next_state_t)

        t = torch.FloatTensor([0.0 if is_terminal else 1.0]).to(self.args.device)
        transition = [self.current_state, act_vector_t, reward_t, next_state_t, t]
        self.memory.add(transition)

        self.current_state = next_state_t

    def end_step(self, reward):
        self.e_reward.append(reward)
        self.update_target()

        # Pre populate memory before replay
        if self.memory.size() >= self.args.batch_size * 2:
            self.replay()

        self.total_steps += 1
        self.current_step += 1
    
    def replay(self):     
        minibatch, idxs, importance_sampling_weight = self.memory.get_batch(self.args.batch_size)

        state_t = torch.stack([x[0] for x in minibatch]).to(self.args.device)
        recorded_action_t = torch.stack([x[1] for x in minibatch])
        reward_t = torch.cat([x[2] for x in minibatch])
        next_state_t = torch.stack([x[3] for x in minibatch])
        done_t = torch.cat([x[4] for x in minibatch])

        # CURIOSITY LOSS
        if self.args.is_curiosity:
            loss_inv, loss_cos = self.get_inverse_and_forward_loss(state_t, next_state_t, recorded_action_t)  
            reward_t += loss_cos.detach() * self.args.curiosity_scale
        
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
        loss.backward(retain_graph=True)
        self.optimizer_agent.step()
        self.optimizer_agent.zero_grad()

        if self.epsilon > self.args.epsilon_floor:
            self.epsilon -= self.args.epsilon_decay
   
    # ====     AGENT INTERNAL STATE    =====       
    def update_target(self):
        if self.args.is_ddqn:
            if self.total_steps % self.args.target_update == 0:
                self.target_model.load_state_dict(self.dqn_model.state_dict())
        
    # ===     DEBUGGING   =====
    def print_debug(self, i_episode, exec_time):
        if self.args.debug:
            dqn_loss = self.loss_dqn[-1] if self.loss_dqn else 0
            ers = sum(self.e_reward)
            info = f"i_episode: {i_episode} | epsilon: {self.epsilon:.4f} |  dqn:  {dqn_loss:.4f} | ers:  {ers:.2f} | time: {exec_time:.2f} | mem: {self.memory.size()}"

            if self.args.prioritized_type != 'random':
                info += f' | per_b: {self.memory.mem.per_b:.2f}'
                
            if self.args.is_curiosity:
                curious_info = f" | n_steps: {self.total_steps} | com: {self.loss_combined[-1]:.4f} | inv: {self.loss_inverse[-1]:.4f} | cos: {self.cos_distance[-1]:.4f}"
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
        d['per_a'] = self.args.per_a
        d['per_b'] = self.args.per_b

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
            next_state_Q_max = torch.gather(next_state_target_val, dim=1, index=next_state_Q_max_idx)
            next_state_Q_max = torch.squeeze(next_state_Q_max, dim=1) 
        else:
            next_state_Q_max, _ = next_state_Q_val.max(dim=1) 

        Q_next_t = reward_t + done_t * self.args.gamma * next_state_Q_max

        Q_cur_t = self.dqn_model(state_t)
        Q_cur_t = Q_cur_t * recorded_action_t 
        Q_cur_t = torch.sum(Q_cur_t, dim=1) # gets rid of zeros

        mse = nn.MSELoss(reduction='none') # REDUCTION PARAMETER!
        loss_dqn = mse(Q_cur_t, Q_next_t)

        # # PER
        if self.args.prioritized_type != 'random':
            loss_dqn = (torch.FloatTensor(importance_sampling_weight).to(self.args.device) * loss_dqn)

            td_errors = torch.abs(Q_next_t - Q_cur_t)
            self.memory.update(to_numpy(td_errors), idxs) 

        return loss_dqn


    def get_inverse_and_forward_loss(self, state_t, next_state_t, recorded_action_t):
        # --------------- INVERSE MODEL -----------------------
        trans = torch.cat((state_t, next_state_t), dim=1)
        pred_action = self.inverse_model(trans)

        loss_inverse = -torch.mean(recorded_action_t * torch.log(pred_action + 1e-12))

        # --------------- FORWARD MODEL / CURIOSITY -------------------------
        cat_t = torch.cat((state_t, recorded_action_t), dim=1)

        pred_next_state_t = self.forward_model(cat_t)
        loss_cos = F.cosine_similarity(pred_next_state_t, next_state_t, dim=1)  
        loss_cos = 1.0 - loss_cos

        # DEBUGGING
        if self.args.debug_features:
            debug_encoded_states(pred_next_state, next_state_t)

        # DEBUGGING
        if self.args.debug_images:
            key = self.args.debug_activations[0]
            debug_sequence(self.states_sequence, self.feature_encoder.encoder.activations[key])
  
        return loss_inverse, loss_cos