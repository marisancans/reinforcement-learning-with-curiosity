import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import logging

def to_numpy(tensor_data):
    return tensor_data.detach().to('cpu').data.numpy()

def init_parameters(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size

        if param.requires_grad == True:
            if len(param.size()) > 1: # is weight or bias
                if 'conv' in name and name.endswith('.weight'):
                    torch.nn.init.kaiming_uniform_(param, mode='fan_out', nonlinearity='relu')
                elif '.bn' in name or '_bn' in name:
                    if name.endswith('.weight'):
                        torch.nn.init.normal_(param, 1.0, 0.02)
                    else:
                        torch.nn.init.constant_(param, 0.0)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
                else: # linear
                    torch.nn.init.xavier_uniform_(param)
            else:
                if 'bias' in name:
                    param.data.zero_()
                else:
                    torch.nn.init.uniform_(param)

    logging.info(f'total_param_size: {total_param_size}')


def normalize_output(output_emb, embedding_norm):
    if embedding_norm == 'l2':
        norm = torch.norm(output_emb.detach(), p=2, dim=1, keepdim=True)
        output_norm = output_emb / norm
    else: # none
        output_norm = output_emb
    return output_norm
