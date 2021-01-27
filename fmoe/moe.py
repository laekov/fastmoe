import math
from torch import nn
import torch
import torch.nn.functional as F

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
    def __init__(self, num_expert=32, d_model=1024, d_hidden=4096, 
            world_size=None, activation=torch.nn.functional.gelu,
            top_k=2, pre_lnorm=False):
        super(FFFN, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.world_size = world_size
        self.activation = activation
        self.top_k = top_k
        self.pre_lnorm = pre_lnorm

        self.htoh4 = FMoE(num_expert, d_model, d_hidden,
                world_size=world_size)
        self.h4toh = FMoE(num_expert, d_hidden, d_model, 
                world_size=world_size)
        self.gate = nn.Linear(d_model, num_expert * world_size)
        self.layer_norm = nn.LayerNorm(d_model)
        self.bias = torch.nn.parameter.Parameter(torch.zeros(d_model,
                dtype=torch.float32)) 

    def forward(self, inp):
        # import pdb; pdb.set_trace()
        residual = inp
        if self.pre_lnorm:
            inp = self.layer_norm(inp)

        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(gate, k=self.top_k, dim=-1,
                largest=True, sorted=False) # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k 
        gate_score = F.softmax(gate_top_k_val, dim=-1).unsqueeze(1) 
        gate_top_k_idx = gate_top_k_idx.view(-1) # (BxLxtop_k)

        inp = inp.view(-1, self.d_model).repeat_interleave(repeats=self.top_k, 
                dim=0) # (BxLxtop_k) x d_model
        x = self.htoh4(inp, gate_top_k_idx)
        x = self.activation(x)
        x = self.h4toh(x, gate_top_k_idx)

        core_out = x.view(-1, self.top_k, self.d_model) # (BxL) x top_k x d_model 
        core_out = torch.bmm(gate_score, core_out) # (BxL) x 1 x d_model
        core_out = core_out.view(residual.size(0), residual.size(1), self.d_model)
        output = core_out + residual

        if not self.pre_lnorm:
            output = self.layer_norm(output)
        return output, self.bias


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
