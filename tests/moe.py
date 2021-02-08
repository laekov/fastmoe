import math
from torch import nn
import torch
import torch.nn.functional as F


class BruteForceMoELinear(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024, 
            world_size=0):
        super(BruteForceMoELinear, self).__init__()
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
            self.weight.data[i] = linear.weight.data
    
    def forward(self, inp, gate):
        gate_long = gate.long()
        batch_size = inp.size(0)
        o = torch.empty(batch_size, self.out_feat, dtype=inp.dtype,
                device=inp.device)
        for i in range(self.num_expert):
            idx = (gate == i)
            x = inp[idx]
            x = x @ self.weight[i].t()
            o[idx] = x
        return o
