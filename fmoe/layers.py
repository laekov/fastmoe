r'''
Layers that FMoE provides to users
'''
import math
import torch
import torch.nn as nn
import numpy as np

from .functions import moe_prepare_forward
from .functions import MOEScatter, MOEGather, MOELinear
from .functions import AllGather
from .gates import NaiveGate


class FMoELinear(nn.Module):
    r'''
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    '''
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024, rank=0):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        r'''
        Initialize the weight as linear layers
        '''
        rng = np.random.default_rng(np.random.randint(2048) + self.rank)

        # copied from torch.nn.init.kaiming_uniform_
        fan = nn.init._calculate_correct_fan(self.weight[0], 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        device = self.weight.device
        dtype = self.weight.dtype
        for i in range(self.num_expert):
            weight = rng.uniform(-bound, bound,
                    size=tuple(self.weight[i].size()))
            self.weight.data[i] = torch.tensor(weight,
                    dtype=dtype, device=device)

    def forward(self, inp, fwd_expert_count):
        r'''
        Call MOE function
        '''
        return MOELinear.apply(inp, self.weight, fwd_expert_count)


def mark_module_parallel_comm(module, comm):
    r'''
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    '''
    for p in module.parameters():
        setattr(p, 'dp_comm', comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size):
    r'''
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
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
    x = expert_fn(x, fwd_expert_count)
    x = MOEGather.apply(
        x, pos, local_expert_count, global_expert_count, inp.shape[0], world_size
    )
    return x


class FMoE(nn.Module):
    r'''
    A general moe implementation that supports an arbitrary module as the expert
    Either `expert` or `expert_fn` is required.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `mp_group` can be a torch's communication group, indicating that model
    parallel is applied across the group, which means that workers in the group
    hold the same copy of the input feature, and demands the same copy of the
    output. FMoE saves computation by slicing the input in the mp group and
    performing all-gather after the MLP computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    * `expert_fn` is specified as a callable object or a function, it will be
    called during forward, giving the input tensor (contiguous) and the array of
    the number of input feature to each expert as input.
    '''
    def __init__(self, num_expert=32, d_model=1024, world_size=1, mp_group=None,
            top_k=2, gate=NaiveGate, expert=None, expert_fn=None):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.mp_group = mp_group
        if mp_group is None:
            self.mp_size = 1
            self.mp_rank = 0
        else:
            self.mp_size = mp_group.size()
            self.mp_rank = mp_group.rank()
        self.top_k = top_k
        self.gate = gate(d_model, num_expert, world_size, top_k)
        if expert_fn is None:
            assert expert is not None, 'Either expert or expert_fn should be set'
            self.experts = [expert(d_model) for _ in range(num_expert)]
            def expert_fn(inp, fwd_expert_count):
                outputs = []
                base_idx = 0
                for i in range(self.num_expert):
                    batch_size = fwd_expert_count[i].item()
                    inp_slice = inp[base_idx:base_idx + batch_size]
                    outputs.append(self.experts[i](inp_slice))
                    base_idx += batch_size
                return torch.cat(outputs, dim=0)
        self.expert_fn = expert_fn

    def mark_parallel_comm(self):
        r'''
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        '''
        if self.experts is not None:
            if self.world_size > self.mp_size:
                comm = 'none'
            else:
                comm = 'dp'
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, 'world')

    def forward(self, inp):
        r'''
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        '''
        if self.mp_size > 1:
            B: int = inp.shape[0]
            local_batch_size = B // self.mp_size
            batch_start = local_batch_size * self.mp_rank
            batch_end = min(batch_start + local_batch_size, B)
            inp = inp[batch_start:batch_end]

        gate_top_k_idx, gate_score = self.gate(inp)
        # to: (BxLxtop_k) x d_model
        inp = inp.repeat_interleave(repeats=self.top_k, dim=0)
        x = _fmoe_general_global_forward(inp, gate_top_k_idx, self.expert_fn,
                self.num_expert, self.world_size)
        # to: (BxL) x top_k x d_model
        x = x.view(-1, self.top_k, self.d_model)
        # to: (BxL) x d_model
        x = torch.bmm(gate_score, x).reshape(-1, self.d_model)

        if self.mp_size > 1:
            x = AllGather.apply(x,
                    self.mp_rank, self.mp_size, self.mp_group)
        return x
