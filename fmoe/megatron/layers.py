r"""
nn modules to replace Megatron's native ones
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fmoe.transformer import FMoETransformerMLP
from .balance import reset_gate_hook
from .balance import generate_megatron_gate_hook


class _FakeMegatronMLP(nn.Module):
    r"""
    A fake mlp without model parallelism for correctness testing
    """

    def __init__(self, args, _):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_hidden_size)
        self.fc2 = nn.Linear(args.hidden_hidden_size, args.hidden_size)

    def forward(self, x):
        r"""
        Directly use GeLU
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, torch.zeros_like(x)


def _megatron_init_method(self, rng, sigma):
    r"""
    Init method based on N(0, sigma).
    Copied from Megatron-LM
    """
    device = self.weight.device
    dtype = self.weight.dtype
    weight = rng.normal(loc=0.0, scale=sigma, size=tuple(self.weight.size()))
    self.weight.data = torch.from_numpy(weight).to(dtype=dtype, device=device)

    if self.bias is not None:
        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()


def _random_init_weight(self, rng):
    r"""
    Copied from torch.nn.init.kaiming_uniform_
    """
    fan = nn.init._calculate_correct_fan(self.weight[0], "fan_in")
    gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    device = self.weight.device
    dtype = self.weight.dtype
    weight = rng.uniform(-bound, bound, size=tuple(self.weight.size()))
    self.weight.data = torch.from_numpy(weight).to(dtype=dtype, device=device)

    if self.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in)
        bias = rng.uniform(-bound, bound, size=tuple(self.bias.size()))
        self.bias.data = torch.from_numpy(bias).to(dtype=dtype, device=device)


class MegatronMLP(FMoETransformerMLP):
    r"""
    Make the FMoETransformerMLP layer that distributes experts across
    communication group `group` to replace the original MLP layer in Megatron.
    """

    def __init__(self, args, layer_idx, gate=None):
        if not args.distributed_experts:
            world_size = 1
            moe_group = None
        else:
            world_size = args.data_parallel_size
            from megatron.mpu import get_data_parallel_group
            moe_group = get_data_parallel_group()

        if not args.balance_strategy or args.balance_strategy == "naive":
            from fmoe.gates import NaiveGate
            gate = NaiveGate
        elif args.balance_strategy == "noisy":
            from fmoe.gates import NoisyGate
            gate = NoisyGate
        elif args.balance_strategy == "gshard":
            from fmoe.gates import GShardGate
            gate = GShardGate
        elif args.balance_strategy == "switch":
            from fmoe.gates import SwitchGate
            gate = SwitchGate
        elif args.balance_strategy == "swipe":
            from fmoe.gates import SwipeGate
            gate = SwipeGate
        elif gate is None:
            assert False, "Undefined balance strategy {}" % (args.balance_strategy)

        super().__init__(
            args.num_experts,
            top_k=args.top_k,
            d_model=args.hidden_size,
            d_hidden=args.hidden_hidden_size,
            world_size=world_size,
            moe_group=moe_group,
            expert_dp_comm="none" if args.distributed_experts else "dp",
            gate_hook=generate_megatron_gate_hook(
                layer_idx, args.num_experts * world_size
            ),
            gate=gate,
        )
        self.hidden_size = args.hidden_size
        if args.distributed_experts:
            self.rank = args.rank
        else:
            self.rank = 0
        self.sigma = args.init_method_std
        self.num_layers = args.num_layers
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Initialize the weight as linear layers.
        As megatron is using fixed random seed for some nasty stuff, an
        additional numpy rng is used.
        """
        rng = np.random.default_rng(np.random.randint(2048) + self.rank)
        _megatron_init_method(self.experts.htoh4, rng, self.sigma)
        std = self.sigma / math.sqrt(2.0 * self.num_layers)
        _megatron_init_method(self.experts.h4toh, rng, std)

    def forward(self, inp):
        from megatron import mpu
        x = super().forward(inp)
        x = mpu.reduce_from_tensor_model_parallel_region(x)
        return (
            x,
            torch.zeros(self.hidden_size, dtype=inp.dtype, device=inp.device),
        )


def fmoefy(
    model,
    num_experts=None,
    distributed_experts=True,
    hidden_hidden_size=None,
    top_k=None,
    gate=None,
):
    r"""
    Replace MLP layers in a transformer-based model in Megatron by MoE.
    * `model` should be a standard Megatron model that has
    `model.language_model.transformer.layers` as transformer layers, which is an
    array of transformer blocks that contain an `mlp` member.
    * `distributed_expert` is set to True if different experts are located in
    different workers. Otherwise, the experts on the workers are identical, and
    they are trained in data-parallel mode. This can be useful when testing on
    small models that do not require high training throughput or large parameter
    capacity.
    Note that pipeline parallel is not supported yet. When distributed experts
    are enabled, their communicator should be Megatron's
    tensor_model_parall_comm x data_parallel_comm, which is not created.
    """
    from megatron import get_args

    args = get_args()

    # Set distributed_experts to None to use default setting in args
    if distributed_experts is not None:
        args.distributed_experts = distributed_experts

    if num_experts is not None:
        args.num_experts = num_experts
    assert (
        "num_experts" in args
    ), "num_experts should be specified in arguments or fmoefy function"

    if top_k is not None:
        args.top_k = top_k
    elif not hasattr(args, "top_k"):
        args.top_k = 2

    args.hidden_hidden_size = hidden_hidden_size

    for idx, l in enumerate(model.language_model.transformer.layers):
        l.mlp = MegatronMLP(args, idx, gate=gate)

    # initialize gate hook
    num_layers = len(model.language_model.transformer.layers)
    reset_gate_hook(num_layers)

    return model
