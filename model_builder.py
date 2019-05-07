import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.torch_utils import init_parameters

import random, re
import numpy as np



# =======           MODEL  BUILDER       =========

class ModelBuilder():
    def __init__(self, args, n_states, n_actions):
        self.args = args
        self.n_states = n_states
        self.n_actions = n_actions
        self.encoder_output_size = 0

    def get_encoder_out(self):
        if self.args.encoder_type == 'simple':
            in_features = self.args.simple_encoder_2_layer_out
        elif self.args.encoder_type == 'conv':
            in_features = self.encoder_output_size
        else:
            in_features = self.n_states
        
        return in_features


    def build_dqn_model(self):

        seq = torch.nn.Sequential(
            nn.Linear(in_features=self.get_encoder_out(), out_features=self.args.dqn_1_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.dqn_1_layer_out, out_features=self.n_actions),
        ).to(self.args.device)

        seq = init_parameters('dqn', seq)
        return seq

    def build_inverse_model(self):
        seq = torch.nn.Sequential(
            nn.Linear(in_features=self.get_encoder_out() * 2, out_features=self.args.inverse_1_layer_out),
            nn.ReLU(),
            nn.Linear(in_features=self.args.inverse_1_layer_out, out_features=self.n_actions),
            #nn.Softmax()
        ).to(self.args.device)

        seq = init_parameters('inverse', seq)
        return seq

    def build_forward_model(self):
        seq = torch.nn.Sequential(
            nn.Linear(in_features=self.get_encoder_out() + self.n_actions, out_features=self.args.forward_1_layer_out), # input actions are one hot encoded
            nn.ReLU(),
            nn.Linear(in_features=self.args.forward_1_layer_out, out_features=self.get_encoder_out()),
        ).to(self.args.device)
        
        seq = init_parameters('forward', seq)
        return seq


# =======           CONV  MODEL       =========

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

class ConvEncoderModule(nn.Module):
    def __init__(self, args, channels):
        super(ConvEncoderModule, self).__init__()
        self.args = args
        self.channels = channels
        self.out_size = 0

        self.init_encoder()

        if self.args.debug_images:
            self.activations = {}
            self.add_hooks()


    def custom_conv(self, original):
        c  = torch.nn.Conv2d(in_channels=self.channels,
            out_channels=original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False
        )
        return c 

    def init_encoder(self):
        pretrained_model = models.densenet161(pretrained=True)

        # preserve trained weights
        features_pretrained = next(iter(pretrained_model.children()))
        conv0 = features_pretrained.conv0
        weights_conv0 = conv0.weight.data

        conv0_new = self.custom_conv(conv0)
       
        # Copy all weights 
        if conv0.in_channels == self.channels:
            conv0_new.weight.data[:] = weights_conv0[:, :, :, :]

        # Copy less than 3 channels
        elif conv0.in_channels > self.channels:
            conv0_new.weight.data[:] = weights_conv0[:, :self.channels, :, :]
       
        else:
            logging.error(f'Densenet expected {conv0.in_channels} channels, got {self.channels} at init_encoder()')
            os._exit(0)

        self.encoder = torch.nn.Sequential()
        
        for name, module in features_pretrained.named_children():
            if name == 'conv0':
                module = conv0_new
            elif name == 'norm5':
                self.out_size = module.num_features
            self.encoder.add_module(name, module)

        self.encoder.add_module('avg_pool', torch.nn.AdaptiveAvgPool2d(output_size=1))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def add_hooks(self):
        # Add hook to desired layer. This should be done with recursion
        if self.args.debug_activations:
            denseblock, denselayer, conv = self.args.debug_activations[0].split()
            for name, module in self.encoder.named_modules():
                if re.sub('denseblock([0-9]*)$',r'\1', name) == denseblock:
                    for namedb, moduledb in module.named_children():
                        if re.sub('denselayer([0-9]*)$',r'\1', namedb) == denselayer:
                            for namel, modulel in moduledb.named_children():
                                if re.sub('conv([0-9]*)$',r'\1', namel) == conv:
                                    key = self.args.debug_activations[0]
                                    modulel.register_forward_hook(self.get_activation(key))

    def forward(self, x):
        features = self.encoder(x)
        #features = nn.functional.tanh(features)
        #features = features.view(-1)
               
        return features


# =======           SIMPLE ENCODER       =========

class SimpleEncoderModule(nn.Module):
    def __init__(self, args, n_states):
        super(SimpleEncoderModule, self).__init__()

        self.linear_1 = nn.Linear(in_features=n_states, out_features=args.simple_encoder_1_layer_out)
        self.relu_1 = nn.LeakyReLU()
        self.linear_2 = nn.Linear(in_features=args.simple_encoder_1_layer_out, out_features=args.simple_encoder_2_layer_out)
        self.tanh_1 = nn.Tanh()

        init_parameters('simple', self)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.relu_1(x1)
        x3 = self.linear_2(x2)
        x4 = self.tanh_1(x3)
        embedding = x4

        # L2 normalization
        norm = torch.norm(embedding.detach(), p=2, dim=1, keepdim=True)
        output_norm = embedding / norm
        
        return output_norm


# =======           FEATURE EXTRACTOR       =========


class FeatureExtractor():
    def __init__(self, args, n_states):
        self.len_layers_rnn = 1  # This can be tested
        self.args = args

        if args.encoder_type == 'conv':
            channels = 1 if args.is_grayscale else 3
            self.encoder = ConvEncoderModule(args, channels).to(args.device)
        else:
            self.encoder = SimpleEncoderModule(args, n_states).to(self.args.device)
               
        # Input size to RNN is output size from encoders or state size 
        if args.encoder_type == 'conv':
            self.encoder_output_size = self.encoder.out_size
        else:
            self.encoder_output_size = self.args.simple_encoder_2_layer_out

        self.fc_hidden_size = self.encoder_output_size # This is blind guess

        self.layer_rnn = torch.nn.LSTM(
            input_size=self.encoder_output_size,
            hidden_size=self.fc_hidden_size,
            num_layers=self.len_layers_rnn,
            batch_first=True
        ).to(self.args.device)

    def reset_hidden(self, batch_size):
        self.hidden_rnn = torch.zeros(self.len_layers_rnn, batch_size, self.fc_hidden_size).to(self.args.device)
        self.state_rnn = torch.zeros(self.len_layers_rnn, batch_size, self.fc_hidden_size).to(self.args.device)

    def extract_features(self, sequece_t, seq_lengths):
        batch_size = sequece_t.shape[0]

        self.reset_hidden(batch_size)

        # ===   CONV   ===
        if self.args.encoder_type == 'conv': 
            frames, channels, height, width = sequece_t.shape[1:]

            # (Batch, Frames, C, H, W) --> (Batch * Frames, C, H, W)
            sequece_t = torch.reshape(sequece_t, shape=(batch_size * frames, channels, height, width))
              
            sequece_t = self.encoder(sequece_t)

            # (Batch * Frames, C, H, W) --> (Batch, Frames, encoder_out)
            sequece_t = torch.reshape(sequece_t, shape=(batch_size, frames, self.encoder.out_size))                       
        # ===   SIMPLE  ===
        else:
            sequece_t = self.encoder(sequece_t)

        # ===   RNN    ===    
        output, hidden = self.layer_rnn(sequece_t, (self.hidden_rnn, self.state_rnn))

        output = output[:, -1:, :] # Take last output
        output = torch.squeeze(output, 1) # Remove frames dimension 
            
        output = output.detach()# Detach, so that replay memory doesnt save computated graphs
        return output



# class DecoderModule(nn.Module):
#      def __init__(self, args, n_states):
#         super(DecoderModule, self).__init__()

#         self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
#         self.conv_2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

#         self.deconv_1 = torch.nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=5)
#         self.deconv_2 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=3, kernel_size=5)


#     def forward(self, x):
#         # Encoder
#         x = self.conv_1.forward(x)
#         x = F.relu(x)
#         x = self.conv_2.forward(x)
#         x = F.relu(x)

#         # Decoder
#         x = self.deconv_1.forward(x)
#         x = F.relu(x)
#         x = self.deconv_2.forward(x)
#         x = F.sigmoid(x)
#         return x

