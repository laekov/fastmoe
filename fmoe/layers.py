from .functions import *
import torch.nn as nn
import torch.nn.functional as F


class FMoELinear(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024):
        super(FMoELinear, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, out_features=self.out_feat)
            self.weight.data[i] = linear.weight.data

    def forward(self, inp, fwd_expert_count):
        return MOELinear.apply(inp, self.weight, fwd_expert_count)


class FMoENaiveGate(nn.Module):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super(FMoENaiveGate, self).__init__()
        self.gate = nn.Linear(d_model, num_expert * world_size)
        self.top_k = top_k

    def forward(self, inp):
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1).unsqueeze(1)
        gate_top_k_idx = gate_top_k_idx.view(-1)  # (BxLxtop_k)

        return gate_top_k_idx, gate_score


def _fmoe_full_forward(inp, gate, linears, activation, num_expert, world_size):
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = moe_prepare_forward(gate, num_expert, world_size)
    x = MOEScatter.apply(
        inp, pos, local_expert_count, global_expert_count, fwd_batch_size, world_size
    )
    for i, l in enumerate(linears):
        if i:
            x = activation(x)
        x = l(x, fwd_expert_count)
    x = MOEGather.apply(
        x, pos, local_expert_count, global_expert_count, inp.shape[0], world_size
    )
    return x


class FMoETransformerMLP(nn.Module):
    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        world_size=1,
        model_parallel_size=1,
        model_parallel_rank=1,
        mp_group=None,
        activation=torch.nn.functional.gelu,
        top_k=2,
        pre_lnorm=False,
    ):
        super(FMoETransformerMLP, self).__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.world_size = world_size
        self.model_parallel_size = model_parallel_size
        self.model_parallel_rank = model_parallel_rank
        self.mp_group = mp_group
        self.activation = activation
        self.pre_lnorm = pre_lnorm
        self.top_k = top_k

        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model)

        self.gate = FMoENaiveGate(d_model, num_expert, world_size, top_k)
        for p in self.gate.parameters():
            setattr(p, 'dp_comm', 'world')

        self.layer_norm = nn.LayerNorm(d_model)
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(d_model, dtype=torch.float32)
        )

    def forward(self, inp: torch.Tensor):
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        if self.model_parallel_size > 1:
            B: int = inp.shape[0]
            local_batch_size = B // self.model_parallel_size
            batch_start = local_batch_size * self.model_parallel_rank
            batch_end = min(batch_start + local_batch_size, B)
            inp = inp[batch_start:batch_end]

        residual = inp
        if self.pre_lnorm:
            inp = self.layer_norm(inp)

        gate_top_k_idx, gate_score = self.gate(inp)

        # to: (BxLxtop_k) x d_model
        inp = inp.repeat_interleave(repeats=self.top_k, dim=0)

        x = _fmoe_full_forward(
            inp,
            gate_top_k_idx,
            [self.htoh4, self.h4toh],
            self.activation,
            self.num_expert,
            self.world_size,
        )

        # to: (BxL) x top_k x d_model
        core_out = x.view(-1, self.top_k, self.d_model)
        # to: (BxL) x 1 x d_model
        core_out = torch.bmm(gate_score, core_out)
        output = core_out.reshape(residual.shape) + residual

        if not self.pre_lnorm:
            output = self.layer_norm(output)

        if self.model_parallel_size > 1:
            world_size = self.model_parallel_size
            tensor_list = [torch.empty_like(output) for _ in range(world_size)]

            torch.distributed.all_gather(tensor_list, output, group=self.mp_group)
            output = torch.cat(tensor_list, dim=1)

        return output.reshape(original_shape), self.bias
