import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import random, re
import numpy as np

class SimpleEncoderModule(nn.Module):
    def __init__(self, args, n_states):
        super(SimpleEncoderModule, self).__init__()

        self.linear_1 = nn.Linear(in_features=n_states, out_features=args.simple_encoder_1_layer_out)
        self.relu_1 = nn.LeakyReLU()
        self.linear_2 = nn.Linear(in_features=args.simple_encoder_1_layer_out, out_features=args.simple_encoder_2_layer_out)
        self.tanh_1 = nn.Tanh()

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.relu_1(x1)
        x3 = self.linear_2(x2)
        x4 = self.tanh_1(x3)
        embedding = x4

        
        # L2 normalization
        norm = torch.norm(embedding.detach(), p=2, dim=1, keepdim=True)
        output_norm = embedding / norm

        c = output_norm.cpu().detach().numpy()
        if np.isnan(c).any():
            a = 1
        
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

        self.dense._modules['classifier'] = nn.Linear(2208, self.args.encoder_layer_out)
        # self.num_features = nn.Sequential(*list(self.dense.children())[0])[-1].num_features # get output feature count from last layer
    

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
