import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.torch_utils import init_parameters, to_numpy

from collections import deque
import random, re, math, cv2
import numpy as np

def build_linear_relu(feature_count, layer_count, first_layer_in, last_layer_out):
    seq = nn.Sequential()

    prev_layer_out = None
    out = feature_count

    for idx in range(layer_count):
        if idx == 0:
            in_features = first_layer_in
        else:
            in_features = prev_layer_out

        if idx == layer_count - 1:
            out = last_layer_out 

        seq.add_module(f"linear_{idx}", torch.nn.Linear(in_features=in_features, out_features=out))

        if idx < layer_count - 1:
            seq.add_module(f"relu_{idx}", nn.ReLU())
            
        prev_layer_out = out
    
    return seq

# ======     DQN    ===========
class DQN_model(nn.Module):
    def __init__(self, feature_count, layer_count, first_layer_in, last_layer_out):
        super(DQN_model, self).__init__()

        self.seq = build_linear_relu(feature_count, layer_count, first_layer_in, last_layer_out)
        init_parameters('dqn', self)

    def forward(self, x):
        out = self.seq(x)
        return out

# ======     INVERSE    ===========
class Inverse_model(nn.Module):
    def __init__(self, feature_count, layer_count, first_layer_in, last_layer_out):
        super(Inverse_model, self).__init__()

        self.seq = build_linear_relu(feature_count, layer_count, first_layer_in, last_layer_out)
        init_parameters('inverse', self)

    def forward(self, x):
        out = self.seq(x)
        out = torch.nn.functional.softmax(out, dim=1)
        return out

# ======     FORWARD    ===========
class Forward_model(nn.Module):
    def __init__(self, feature_count, layer_count, first_layer_in, last_layer_out):
        super(Forward_model, self).__init__()

        self.seq = build_linear_relu(feature_count, layer_count, first_layer_in, last_layer_out)
        init_parameters('forward', self)

    def forward(self, x):
        out = self.seq(x)
        return out

# =======           MODEL  BUILDER       =========

class ModelBuilder():
    def __init__(self, args, n_states, n_actions):
        self.args = args
        self.n_states = n_states
        self.n_actions = n_actions
        self.encoder_output_size = 0

    def get_encoder_out(self):
        if self.args.encoder_type != 'nothing':
            in_features = self.args.encoding_size
        else:
            in_features = self.n_states
        
        return in_features

    def build_dqn_model(self):
        return DQN_model(feature_count=self.args.models_layer_features,
                         layer_count=self.args.models_layer_count,
                         first_layer_in=self.get_encoder_out(), 
                         last_layer_out=self.n_actions
                         ).to(self.args.device)

    def build_inverse_model(self):
        return Inverse_model(feature_count=self.args.models_layer_features,
                             layer_count=self.args.models_layer_count,
                             first_layer_in=self.get_encoder_out() * 2,
                             last_layer_out=self.n_actions
                             ).to(self.args.device)

    def build_forward_model(self):
        return Forward_model(feature_count=self.args.models_layer_features,
                             layer_count=self.args.models_layer_count,
                             first_layer_in=self.get_encoder_out() + self.n_actions,
                             last_layer_out=self.get_encoder_out()
                            ).to(self.args.device)
        
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

class ConvEncoder(nn.Module):
    def __init__(self, args, channels):
        super(ConvEncoder, self).__init__()
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

        self.encoder_out = torch.nn.Linear(in_features=self.out_size, out_features=self.args.conv_encoder_layer_out)

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

# =======          CONV DECODER       =========
class ConvDecoder(nn.Module):
    def __init__(self, args, encoder_features, original_shape):
        super(ConvDecoder, self).__init__()
        h, w, c = original_shape

        if args.encoder_type == 'conv':
            channels = 1 if args.is_grayscale else 3
      

        self.c1 = nn.ConvTranspose2d(in_channels=2208, out_channels=1104, kernel_size=3) # --> (3, 3)
        self.b1 = nn.BatchNorm2d(num_features=1104)
        self.c2 = nn.ConvTranspose2d(in_channels=1104, out_channels=552, kernel_size=3) # --> (5, 5)
        self.b2 = nn.BatchNorm2d(num_features=552)
        self.c3 = nn.ConvTranspose2d(in_channels=552, out_channels=271, kernel_size=3) # --> (7, 7)
        self.b3 = nn.BatchNorm2d(num_features=271)
        self.c4 = nn.ConvTranspose2d(in_channels=271, out_channels=136, kernel_size=3, padding=2, stride=2) # --> (11, 11)
        self.b4 = nn.BatchNorm2d(num_features=136)
        self.c5 = nn.ConvTranspose2d(in_channels=136, out_channels=68, kernel_size=3, padding=2, stride=2) # --> (19, 19)
        self.b5 = nn.BatchNorm2d(num_features=68)
        self.c6 = nn.ConvTranspose2d(in_channels=68, out_channels=34, kernel_size=3, padding=2, stride=2) # --> (35, 35)
        self.b6 = nn.BatchNorm2d(num_features=34)
        self.c7 = nn.ConvTranspose2d(in_channels=34, out_channels=16, kernel_size=2, padding=2, stride=2) # --> (66, 66)
        self.b7 = nn.BatchNorm2d(num_features=16)
        self.c8 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, padding=2, stride=2) # --> (128, 128)
        self.b8 = nn.BatchNorm2d(num_features=8)
        self.c9 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, padding=2, stride=2) # --> (252, 252)
        self.b9 = nn.BatchNorm2d(num_features=4)
        self.c10 = nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=2, padding=2, stride=2) # --> (500, 500)
        self.b10 = nn.BatchNorm2d(num_features=3)
        
        init_parameters('deconv', self)

    def forward(self, x):   
        x = self.c1(x)
        x = F.relu(x)
        x = self.b1(x)

        x = self.c2(x)
        x = F.relu(x)
        x = self.b2(x)

        x = self.c3(x)
        x = F.relu(x)
        x = self.b3(x)

        x = self.c4(x)
        x = F.relu(x)
        x = self.b4(x)

        x = self.c5(x)
        x = F.relu(x)
        x = self.b5(x)

        x = self.c6(x)
        x = F.relu(x)
        x = self.b6(x)

        x = self.c7(x)
        x = F.relu(x)
        x = self.b7(x)

        x = self.c8(x)
        x = F.relu(x)
        x = self.b8(x)

        x = self.c9(x)
        x = F.relu(x)
        x = self.b9(x)

        x = self.c10(x)
        x = F.relu(x)
        x = self.b10(x)
        return x



# =======           SIMPLE ENCODER       =========z
class SimpleEncoder(nn.Module):
    def __init__(self, args, n_states):
        super(SimpleEncoder, self).__init__()

        self.seq = nn.Sequential()

        prev_layer_out = None

        for idx, out in enumerate(args.simple_encoder_layers):
            if idx == 0:
                in_features = n_states
            else:
                in_features = prev_layer_out
            
            self.seq.add_module(f"linear_{idx}", torch.nn.Linear(in_features=in_features, out_features=out))
            self.seq.add_module(f"relu_{idx}", nn.ReLU())
            
            prev_layer_out = out

        init_parameters('simple encoder', self)

    def forward(self, x):
        x = x.squeeze(dim=0) # squeeze batch (always must have to be batch size = 1) will use timesteps as if batch

        if x.size(0) == 1: # if seq == 1 only single sample
            out = x
            for name, func in self.seq.named_children():
                if "_bn" not in name:
                    out = func(out)
            embedding = out
        else:
            embedding = self.seq(x)
        embedding = embedding.unsqueeze(dim=0)

        return embedding


# =======           SIMPLE ENCODER       =========
class SimpleDecoder(nn.Module):
    def __init__(self, args, first_layer_in, last_layer_out):
        super(SimpleDecoder, self).__init__()

        self.seq = nn.Sequential()
        self.args = args

        for idx, out in enumerate(reversed(args.simple_encoder_layers)):
            if idx == 0:
                in_features = first_layer_in
            else:
                in_features = prev_layer_out

            if idx == len(args.simple_encoder_layers) - 1:
                out = last_layer_out 
            
            self.seq.add_module(f"linear_{idx}", torch.nn.Linear(in_features=in_features, out_features=out))
            #self.seq.add_module(f"layer_{idx}_bn", torch.nn.BatchNorm1d(num_features=out))
            
            if idx < len(args.simple_encoder_layers) - 1:
                self.seq.add_module(f"relu_{idx}", nn.ReLU())
            
            prev_layer_out = out

        init_parameters('simple decoder', self)

    def forward(self, x):
        x = self.seq(x)
        return x


# =======           FEATURE EXTRACTOR       =========
class FeatureEncoder():
    def __init__(self, args, n_states):
        self.rnn_layers = 1  # This can be tested
        self.args = args

        if args.encoder_type == 'conv':
            channels = 1 if args.is_grayscale else 3
            self.encoder = ConvEncoder(args, channels).to(self.args.device)
        else:
            self.encoder = SimpleEncoder(args, n_states).to(self.args.device)
               
        # Input size to RNN is output size from encoders or state size 
        if args.encoder_type == 'conv':
            self.encoder_output_size = self.encoder.out_size
        else:
            self.encoder_output_size = self.args.simple_encoder_layers[-1]

        self.fc_hidden_size = self.args.encoding_size

        self.layer_rnn = torch.nn.GRU(
            input_size=self.encoder_output_size,
            hidden_size=self.fc_hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True
        ).to(self.args.device)

    def reset_hidden(self, batch_size):
        self.hidden_rnn = torch.zeros(self.rnn_layers, batch_size, self.fc_hidden_size).to(self.args.device)
        #self.state_rnn = torch.zeros(self.rnn_layers, batch_size, self.fc_hidden_size).to(self.args.device)

    def extract_features(self, sequece_t):
        sequece_t = sequece_t.to(self.args.device)
        batch_size = sequece_t.shape[0]

        self.reset_hidden(batch_size)

        # ===   CONV   ===
        if self.args.encoder_type == 'conv': 
            frames, channels, height, width = sequece_t.shape[1:] #(B, Frames, C, H, W)

            # (Batch, Frames, C, H, W) --> (Batch * Frames, C, H, W)
            sequece_t = torch.reshape(sequece_t, shape=(batch_size * frames, channels, height, width))
              
            sequece_t = self.encoder(sequece_t)

            # (Batch * Frames, C, H, W) --> (Batch, Frames, encoder_out)
            sequece_t = torch.reshape(sequece_t, shape=(batch_size, frames, self.encoder.out_size))                       
        # ===   SIMPLE  ===
        else:
            sequece_t = self.encoder(sequece_t)

        # ===   RNN    ===    
        output, hidden = self.layer_rnn.forward(sequece_t, self.hidden_rnn)

        output_last = output[:, -1:, :] # Take last output

        embedding = F.tanh(output_last)
        embedding = torch.squeeze(embedding, 1) # Remove frames dimension

        #L2 normalization
        norm = torch.norm(embedding.detach(), p=2, dim=1, keepdim=True)
        if torch.sum(norm) != 0:
            embedding = embedding / norm

        return embedding



class FeatureDecoder():
    def __init__(self, args, encoder_out, n_states, original_shape):
        if args.encoder_type == 'simple':
            self.decoder = SimpleDecoder(args, first_layer_in=encoder_out, last_layer_out=n_states).to(args.device)
        else:
            self.decoder = ConvDecoder(args, encoder_features=encoder_out, original_shape=original_shape).to(args.device)

    def decode_simple_features(self, h_vector):
        h_vector = h_vector.unsqueeze(0)
        pred = self.decoder.forward(h_vector)
        pred = pred.squeeze(0)
        return pred

    def decode_conv_features(self, h_vector):
        feature_count = h_vector.shape[0]
        h_vector = h_vector.view(1, feature_count, 1, 1)
        pred = self.decoder.forward(h_vector)

        return pred

        

