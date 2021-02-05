r'''
Layers that FMoE provides to users
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import moe_prepare_forward
from .functions import MOEScatter, MOEGather, MOELinear
from .functions import AllGather


class FMoELinear(nn.Module):
    r'''
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    '''
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        r'''
        Initialize the weight as linear layers
        '''
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat,
                    out_features=self.out_feat)
            self.weight.data[i] = linear.weight.data

    def forward(self, inp, fwd_expert_count):
        r'''
        Call MOE function
        '''
        return MOELinear.apply(inp, self.weight, fwd_expert_count)


class FMoENaiveGate(nn.Module):
    r'''
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indecies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    '''
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_expert * world_size)
        self.top_k = top_k

    def forward(self, inp):
        r'''
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        '''
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
    r'''
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the MLP of the experts by applying MoELinear and the activation in
    turns.
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    '''
    (
        pos, local_expert_count, global_expert_count, fwd_expert_count,
        fwd_batch_size
    ) = moe_prepare_forward(gate, num_expert, world_size)
    x = MOEScatter.apply(
        inp, pos, local_expert_count, global_expert_count, fwd_batch_size,
        world_size
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
    r'''
    A complete MoE MLP module in a Transformer block.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `mp_group` can be a torch's communication group, indicating that model
    parallel is applied across the group, which means that workers in the group
    hold the same copy of the input feature, and demands the same copy of the
    output. FMoE saves computation by slicing the input in the mp group and
    performing all-gather after the MLP computation.
    * `activation` is the activation function to be used in MLP in each expert.
    * `top_k` stands for the number of experts each token is going to.
    '''
    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        world_size=1,
        mp_group=None,
        activation=torch.nn.functional.gelu,
        top_k=2,
        pre_lnorm=False
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.world_size = world_size
        self.mp_group = mp_group
        if mp_group is None:
            self.mp_size = 1
            self.mp_rank = 0
        else:
            self.mp_size = mp_group.size()
            self.mp_rank = mp_group.rank()
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
        r'''
        The FMoETransformerMLP module automatically performs reshape and layer
        normalization. The score of the selected gate given by the expert is
        multiplied to the experts' output tensors as a weight.
        '''
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        if self.mp_size > 1:
            B: int = inp.shape[0]
            local_batch_size = B // self.mp_size
            batch_start = local_batch_size * self.mp_rank
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

        if self.mp_size > 1:
            output = AllGather.apply(output,
                    self.mp_rank, self.mp_size, self.mp_group)

        return output.reshape(original_shape), self.bias
