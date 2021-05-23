r"""
Layers that FMoE provides to users
"""
import torch
import torch.nn as nn

from .functions import prepare_forward
from .functions import MOEScatter, MOEGather, MOELinear
from .functions import AllGather, Slice
from .gates import NaiveGate


class FMoELinear(nn.Module):
    r"""
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(
        self,
        num_expert: int,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        rank: int = 0,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_expert, out_feat))
        else:
            self.register_parameter("bias", None)

    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """
        x = MOELinear.apply(inp, fwd_expert_count, self.weight, self.bias)
        return x

    def extra_repr(self) -> str:
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )


def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]
    x = MOEScatter.apply(
        inp, pos // topk,
        local_expert_count, global_expert_count, fwd_batch_size, world_size
    )
    x = expert_fn(x, fwd_expert_count)

    out_batch_size = inp.shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    x = MOEGather.apply(
        x, pos,
        local_expert_count, global_expert_count,
        out_batch_size, world_size
    )
    return x


class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
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
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
    ):
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
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model)
                for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
        self.gate = gate(d_model, num_expert, world_size, top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict

    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
        outputs = []
        base_idx = 0
        for i in range(self.num_expert):
            batch_size = fwd_expert_count[i].item()
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "world")

    def forward(self, inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        if self.mp_size > 1:
            inp = Slice.apply(inp, self.mp_rank, self.mp_size, self.mp_group)

        gate_top_k_idx, gate_score = self.gate(inp)

        # delete masked tensors
        if self.mask != None and self.mask_dict != None:
            mask = self.mask.view(-1)
            # to: (BxL') x d_model
            inp = inp[mask == 0, :]
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        fwd = _fmoe_general_global_forward(
            inp,
            gate_top_k_idx,
            self.expert_fn, self.num_expert, self.world_size
        )

        # recover deleted tensors
        if self.mask != None and self.mask_dict != None:
            # to: (BxL') x top_k x d_model
            fwd = fwd.view(-1, self.top_k, self.d_model)
            # to: (BxL) x top_k x d_model
            x = torch.zeros(mask.shape[0], self.top_k, self.d_model)
            # recover
            x[mask == 0] = fwd
            for k, v in self.mask_dict.items():
                x[mask == k] = v
        else:
            x = fwd.view(-1, self.top_k, self.d_model)

        gate_score = gate_score.view(x.shape[0], 1, self.top_k)
        x = torch.bmm(gate_score, x).reshape(-1, self.d_model)

        if self.mp_size > 1:
            x = AllGather.apply(x, self.mp_rank, self.mp_size, self.mp_group)
        return x
