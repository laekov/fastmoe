import math
from torch import nn
import torch

from .moe_function import moe


class FMoE(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024,
            world_size=None):
        super(FMoE, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.world_size = world_size
        self.weight = nn.Parameter(
            torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, out_features=self.out_feat)
            self.weight.data[i] = linear.weight.data

    def forward(self, inp, gate):
        return moe(inp, gate.int(), self.weight, self.world_size)


class FFFN(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, hidden_feat=4096, 
            out_feat=1024, world_size=None, activation=torch.nn.functional.gelu):
        super(FFFN, self).__init__()
        self.htoh4 = FMoE(num_expert, in_feat, hidden_feat,
                world_size=world_size)
        self.activation = activation
        self.h4toh = FMoE(num_expert, hidden_feat, out_feat, 
                world_size=world_size)

    def forward(self, inp, gate):
        x = self.htoh4(inp)
        x = self.activation(x)
        x = self.h4toh(x)
        return x


class BruteForceMoE(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024, 
            world_size=0):
        super(BruteForceMoE, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(
            torch.Tensor(num_expert * world_size, out_feat, in_feat))
        self.reset_parameters()


    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, 
                    out_features=self.out_feat)
            # print(linear.weight.shape)
            self.weight.data[i] = linear.weight.data
    
    def forward(self, inp, gate):
        gate_long = gate.long()
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.out_feat))
        for i in range(batch_size):
            x[i] = inp[i] @ self.weight[gate_long[i]].t()
        return x
