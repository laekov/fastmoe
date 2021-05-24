import math
from torch import nn
import torch


class BruteForceMoELinear(nn.Module):
    def __init__(
        self,
        activation,
        num_expert=32,
        d_model=1024,
        d_hidden=2048,
        world_size=1,
        top_k=2,
    ):
        super(BruteForceMoELinear, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.activation = activation
        self.weight_htoh4 = nn.Parameter(
            torch.Tensor(num_expert * world_size, d_hidden, d_model)
        )
        self.bias_htoh4 = nn.Parameter(torch.Tensor(num_expert * world_size, d_hidden))
        self.weight_h4toh = nn.Parameter(
            torch.Tensor(num_expert * world_size, d_model, d_hidden)
        )
        self.bias_h4toh = nn.Parameter(torch.Tensor(num_expert * world_size, d_model))
        self.top_k = top_k

    def forward(self, inp, gate_idx, gate_score):
        inp = inp.repeat_interleave(repeats=self.top_k, dim=0)
        gate_long = gate_idx.long().view(-1)
        batch_size = inp.size(0)
        o = torch.empty(batch_size, self.d_model, dtype=inp.dtype, device=inp.device)
        for i in range(self.weight_htoh4.shape[0]):
            idx = gate_long == i
            x = inp[idx]
            x = x @ self.weight_htoh4[i].t()
            x = x + self.bias_htoh4[i]
            x = self.activation(x)
            x = x @ self.weight_h4toh[i].t()
            x = x + self.bias_h4toh[i]
            o[idx] = x
        gate_score = gate_score.unsqueeze(1)

        x = torch.bmm(gate_score, o.view(-1, self.top_k, self.d_model)).reshape(
            -1, self.d_model
        )
        return x


class BruteForceMoE(nn.Module):
    def __init__(self, expert, num_expert=32, d_model=1024, world_size=1, top_k=2):
        super(BruteForceMoE, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.top_k = top_k
        if type(expert) is list:
            self.experts = [e(d_model) for e in expert]
            self.num_expert = num_expert = len(expert)
        else:
            self.experts = [expert(d_model) for _ in range(num_expert * world_size)]

    def forward(self, inp, gate_idx, gate_score):
        inp = inp.repeat_interleave(repeats=self.top_k, dim=0)
        gate_long = gate_idx.long().view(-1)
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.d_model))
        for i in range(batch_size):
            x[i] = self.experts[gate_long[i]](inp[i])
        gate_score = gate_score.unsqueeze(1)
        x = torch.bmm(gate_score, x.view(-1, self.top_k, self.d_model)).reshape(
            -1, self.d_model
        )
        return x


class NaiveExpert(nn.Module):
    def __init__(self, d_model):
        super(NaiveExpert, self).__init__()
        self.linear = nn.Linear(d_model, d_model).cuda()

    def forward(self, x):
        return self.linear(x)


class LinearExpert(nn.Module):
    def __init__(self, d_model):
        super(LinearExpert, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model),
        ).cuda()

    def forward(self, x):
        return self.model(x)
