import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.torch_utils import init_parameters

import random, re
import numpy as np

def build_linear_relu(args, first_layer_in, last_layer_out, last_layer_activation='relu'):
    seq = nn.Sequential()

    prev_layer_out = None
    out = args.models_layer_features

    for idx in range(args.models_layer_count):
        if idx == 0:
            in_features = first_layer_in
        else:
            in_features = prev_layer_out

        if idx == args.models_layer_count - 1:
            out = last_layer_out 

        seq.add_module(f"linear_{idx}", torch.nn.Linear(in_features=in_features, out_features=out))

        if idx == args.models_layer_count - 1 and last_layer_activation == 'softmax':
            seq.add_module(f"softmax_{idx}", nn.Softmax(dim=1))
        else:
            seq.add_module(f"relu_{idx}", nn.ReLU())
        
        prev_layer_out = out
    
    return seq



# ======     DQN    ===========
class DQN_model(nn.Module):
    def __init__(self, args, first_layer_in, last_layer_out):
        super(DQN_model, self).__init__()

        self.seq = build_linear_relu(args, first_layer_in, last_layer_out)
        init_parameters('dqn', self.seq)

    def forward(self, x):
        out = self.seq(x)
        return out

# ======     INVERSE    ========
class Inverse_model(nn.Module):
    def __init__(self, args, first_layer_in, last_layer_out):
        super(Inverse_model, self).__init__()

        self.seq = build_linear_relu(args, first_layer_in, last_layer_out)
        init_parameters('inverse', self.seq)

    def forward(self, x):
        out = self.seq(x)
        return out


# ======     FORWARD    ========
class Forward_model(nn.Module):
    def __init__(self, args, first_layer_in, last_layer_out):
        super(Forward_model, self).__init__()

        self.seq = build_linear_relu(args, first_layer_in,last_layer_out, last_layer_activation='softmax')
        init_parameters('forward', self.seq)

    def forward(self, x):
        out = self.seq(x)
        return out

    # def get_encoder_out(self):
    #     if self.args.encoder_type == 'simple':
    #         in_features = self.args.models_layer_features
    #     elif self.args.encoder_type == 'conv':
    #         in_features = self.encoder_output_size
    #     else:
    #         in_features = self.n_states
        
    #     return in_features


        



    # ======     FORWARD    ========

    def build_forward_model(self):
        seq = self.build_linear_relu(first_layer_in=self.get_encoder_out() + self.n_actions, last_layer_out=self.get_encoder_out())
        
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

        self.seq = nn.Sequential()

        prev_layer_out = None
        out = args.models_layer_features

        for idx in range(args.models_layer_count):
            if idx == 0:
                in_features = n_states
            else:
                in_features = prev_layer_out
            
            self.seq.add_module(f"linear_{idx}", torch.nn.Linear(in_features=in_features, out_features=out))
            self.seq.add_module(f"layer_{idx}_bn", torch.nn.BatchNorm1d(num_features=out))
            
            if idx == args.models_layer_count - 1:
                self.seq.add_module("tanh", nn.Tanh())
            else:
                self.seq.add_module(f"relu_{idx}", nn.ReLU())
            prev_layer_out = out

        init_parameters('simple', self)

    def forward(self, x):
        x = x.squeeze(dim=0)

        if x.size(0) == 1: # if seq == 1 only single sample
            out = x
            for name, func in self.seq.named_children():
                if "_bn" not in name:
                    out = func(out)
            embedding = out
        else:
            embedding = self.seq(x)

        embedding = embedding.unsqueeze(dim=0)

        # L2 normalization
        # norm = torch.norm(embedding.detach(), p=2, dim=1, keepdim=True)
        # if torch.sum(norm) != 0:
        #     embedding = embedding / norm
        
        return embedding


# =======           FEATURE EXTRACTOR       =========


class FeatureExtractor():
    def __init__(self, args, n_states):
        self.len_layers_rnn = 1  # This can be tested
        self.args = args

        if args.encoder_type == 'conv':
            channels = 1 if args.is_grayscale else 3
            self.encoder = ConvEncoderModule(args, channels).to(args.device)
        else:
            self.encoder = SimpleEncoderModule(args, n_states).to(args.device)
               
        # Input size to RNN is output size from encoders or state size 
        encoder_output_size = args.models_layer_features
        self.fc_hidden_size = encoder_output_size # This is blind guess

        self.layer_rnn = torch.nn.LSTM(
            input_size=encoder_output_size,
            hidden_size=self.fc_hidden_size,
            num_layers=self.len_layers_rnn,
            batch_first=True
        ).to(self.args.device)

    def reset_hidden(self, batch_size):
        # (num_layers * num_directions, batch, hidden_size)
        self.hidden_rnn = torch.zeros(self.layer_rnn.num_layers, batch_size, self.fc_hidden_size).to(self.args.device)
        self.state_rnn = torch.zeros(self.layer_rnn.num_layers, batch_size, self.fc_hidden_size).to(self.args.device)

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

