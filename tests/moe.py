import math
from torch import nn
import torch


class BruteForceMoELinear(nn.Module):
    def __init__(self, activation, num_expert=32, d_model=1024, world_size=1, top_k=2):
        super(BruteForceMoELinear, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.activation = activation
        self.weight_htoh4 = nn.Parameter(
            torch.Tensor(num_expert * world_size, d_model * 4, d_model)
        )
        self.weight_h4toh = nn.Parameter(
            torch.Tensor(num_expert * world_size, d_model, d_model * 4)
        )
        self.top_k = top_k

    def forward(self, inp, gate_idx, gate_score):
        gate_long = gate_idx.long()
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.d_model))
        for i in range(batch_size):
            t = inp[i] @ self.weight_htoh4[gate_long[i]].t()
            t = self.activation(t)
            x[i] = t @ self.weight_h4toh[gate_long[i]].t()
        x = torch.bmm(gate_score, x.view(-1, self.top_k, self.d_model)).reshape(
            -1, self.d_model
        )
        return x
