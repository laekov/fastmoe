import sys
from collections import OrderedDict
from typing import List, Type, Union

import pytest
import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from fmoe.gates import NaiveGate
from fmoe.layers import FMoE
from fmoe.transformer import _Expert
from fmoe.distributed import DistributedGroupedDataParallel as LocalDDP
from fmoe.megatron.layers import _megatron_init_method
from moe import BruteForceMoELinear, BruteForceMoE, NaiveExpert, LinearExpert


def _perform_forward(
    moe: nn.Module, moe_raw: nn.Module, batch_size, d_model, top_k, rank, mp_group, data_type='torch.FloatTensor'
):
    moe.zero_grad()
    moe_raw.zero_grad()

    inp = torch.rand(batch_size, d_model).type(data_type).cuda()
        
    if mp_group is not None:
        group_sender = rank // mp_group.size() * mp_group.size()
        torch.distributed.broadcast(inp, group_sender, group=mp_group)
        torch.distributed.broadcast(
            moe.gate.gate.weight.data, group_sender, group=mp_group
        )
        torch.distributed.broadcast(
            moe.gate.gate.bias.data, group_sender, group=mp_group
        )

    inp_raw = inp.clone()
    inp.requires_grad = True

    inp_raw.requires_grad = True
    gate_idx, gate_score = moe.gate(inp_raw)
    moe_out = moe(inp)
    raw_out = moe_raw(inp_raw, gate_idx, gate_score)

    raw_out.mean().backward()
    moe_out.mean().backward()

    return moe_out, raw_out, inp.grad, inp_raw.grad


def _assert_numerical(names, moe_out_list, raw_out_list, rank, precision=1e-3):
    for name, mo, ro in zip(names, moe_out_list, raw_out_list):
        err = (mo - ro).abs().max()
        print("Rank {} {} abs err {}".format(rank, name, err))
        if err > precision:
            sys.stderr.write(f"=========== {name} moe out ==============\n")
            sys.stderr.write("{}\n".format(mo))
            sys.stderr.write(f"=========== {name} raw out ==============\n")
            sys.stderr.write("{}\n".format(ro))
            sys.stderr.write(f"=========== {name} diff ==============\n")
            sys.stderr.write("{}\n{}\n".format((mo - ro).abs(), err))
            assert False


class MyMoE(FMoE):
    def __init__(
        self, num_expert, d_model, d_hidden, world_size, mp_group, top_k, activation
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            gate=NaiveGate,
            world_size=world_size,
            mp_group=mp_group,
            top_k=top_k,
        )
        self.experts = _Expert(num_expert, d_model, d_hidden, activation)

        rng = np.random.default_rng(1234)
        _megatron_init_method(self.experts.htoh4, rng, 1.0)
        _megatron_init_method(self.experts.h4toh, rng, 1.0)


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("d_hidden", [32])
@pytest.mark.parametrize("rank", [0])
@pytest.mark.parametrize("world_size", [1])
@pytest.mark.parametrize("mp_group", [None])
@pytest.mark.parametrize("dp_group", [None])
@pytest.mark.parametrize("world_group", [None])
@pytest.mark.parametrize("data_type", ['torch.FloatTensor', 'torch.DoubleTensor', 'torch.HalfTensor'])
def test_fmoe_linear(
    num_expert,
    top_k,
    batch_size,
    d_model,
    d_hidden,
    rank,
    world_size,
    mp_group,
    dp_group,
    world_group,
    data_type,
    activation=torch.nn.functional.gelu,
):
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    moe = MyMoE(
        num_expert, d_model, d_hidden, world_size, mp_group, top_k, activation
    ).type(data_type).cuda()

    moe_raw = BruteForceMoELinear(
        activation=activation,
        num_expert=num_expert,
        d_model=d_model,
        d_hidden=d_hidden,
        world_size=world_size,
        top_k=top_k,
    ).type(data_type).cuda()

    if world_size == 1:
        moe_raw.weight_htoh4.data = moe.experts.htoh4.weight.data.clone()
        moe_raw.bias_htoh4.data = moe.experts.htoh4.bias.data.clone()
        moe_raw.weight_h4toh.data = moe.experts.h4toh.weight.data.clone()
        moe_raw.bias_h4toh.data = moe.experts.h4toh.bias.data.clone()
    else:
        weight_htoh4_array = [
            torch.empty_like(moe.experts.htoh4.weight.data) for _ in range(world_size)
        ]
        bias_htoh4_array = [
            torch.empty_like(moe.experts.htoh4.bias.data) for _ in range(world_size)
        ]
        torch.distributed.all_gather(weight_htoh4_array, moe.experts.htoh4.weight.data)
        torch.distributed.all_gather(bias_htoh4_array, moe.experts.htoh4.bias.data)
        moe_raw.weight_htoh4.data = torch.cat(weight_htoh4_array, dim=0)
        moe_raw.bias_htoh4.data = torch.cat(bias_htoh4_array, dim=0)

        weight_h4toh_array = [
            torch.empty_like(moe.experts.h4toh.weight.data) for _ in range(world_size)
        ]
        bias_h4toh_array = [
            torch.empty_like(moe.experts.h4toh.bias.data) for _ in range(world_size)
        ]
        torch.distributed.all_gather(weight_h4toh_array, moe.experts.h4toh.weight.data)
        torch.distributed.all_gather(bias_h4toh_array, moe.experts.h4toh.bias.data)
        moe_raw.weight_h4toh.data = torch.cat(weight_h4toh_array, dim=0)
        moe_raw.bias_h4toh.data = torch.cat(bias_h4toh_array, dim=0)

    moe_out, raw_out, moe_grad_in, raw_grad_in = _perform_forward(
        moe, moe_raw, batch_size, d_model, top_k, rank, mp_group, data_type=data_type
    )

    moe_out_list = (
        moe_out,
        moe_grad_in,
        moe.experts.htoh4.weight.grad,
        moe.experts.h4toh.weight.grad,
        moe.experts.htoh4.bias.grad,
        moe.experts.h4toh.bias.grad,
    )
    raw_out_list = (
        raw_out,
        raw_grad_in,
        moe_raw.weight_htoh4.grad,
        moe_raw.weight_h4toh.grad,
        moe_raw.bias_htoh4.grad,
        moe_raw.bias_h4toh.grad,
    )

    if world_size > 1:
        _, __, htoh4_w_grad, h4toh_w_grad, htoh4_b_grad, h4toh_b_grad = raw_out_list
        torch.distributed.all_reduce(htoh4_w_grad)
        torch.distributed.all_reduce(h4toh_w_grad)
        torch.distributed.all_reduce(htoh4_b_grad)
        torch.distributed.all_reduce(h4toh_b_grad)
        mp_size = mp_group.size() if mp_group else 1
        htoh4_w_grad = (
            htoh4_w_grad[rank * num_expert : (rank + 1) * num_expert] / mp_size
        )
        h4toh_w_grad = (
            h4toh_w_grad[rank * num_expert : (rank + 1) * num_expert] / mp_size
        )
        htoh4_b_grad = (
            htoh4_b_grad[rank * num_expert : (rank + 1) * num_expert] / mp_size
        )
        h4toh_b_grad = (
            h4toh_b_grad[rank * num_expert : (rank + 1) * num_expert] / mp_size
        )
        raw_out_list = _, __, htoh4_w_grad, h4toh_w_grad, htoh4_b_grad, h4toh_b_grad

    names = [
        "output",
        "input grad",
        "htoh4 weight grad",
        "h4toh weight grad",
        "htoh4 bias grad",
        "h4toh bias grad",
    ]


    precision = 5e-1 if data_type == 'torch.HalfTensor' else 1e-3
        
    _assert_numerical(names, moe_out_list, raw_out_list, rank, precision=precision)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("expert", [NaiveExpert, LinearExpert])
@pytest.mark.parametrize("rank", [0])
@pytest.mark.parametrize("world_size", [1])
@pytest.mark.parametrize("mp_group", [None])
@pytest.mark.parametrize("dp_group", [None])
@pytest.mark.parametrize("world_group", [None])
def test_fmoe(
    batch_size,
    num_expert,
    d_model,
    top_k,
    expert: Union[Type[nn.Module], str],
    rank,
    world_size,
    mp_group,
    dp_group,
    world_group,
):
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    if isinstance(expert, str):
        expert = globals()[expert]

    moe = FMoE(
        num_expert=num_expert,
        d_model=d_model,
        gate=NaiveGate,
        world_size=world_size,
        mp_group=mp_group,
        expert=expert,
        top_k=top_k,
    ).cuda()

    moe_raw = BruteForceMoE(
        expert=expert,
        num_expert=num_expert,
        d_model=d_model,
        world_size=world_size,
        top_k=top_k,
    ).cuda()

    if world_size == 1:
        for expert_moe, expert_raw in zip(moe.experts, moe_raw.experts):
            for para_moe, para_raw in zip(
                expert_moe.parameters(), expert_raw.parameters()
            ):
                para_raw.data = para_moe.data.clone()
    else:
        assert len(moe.experts) >= 1
        for idx, para in enumerate(moe.experts[0].parameters()):
            para_tensor = torch.cat(
                [list(expert.parameters())[idx].unsqueeze(0) for expert in moe.experts]
            )
            para_array = [torch.empty_like(para_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(para_array, para_tensor)
            para_tensor_gathered = torch.cat(para_array, dim=0)
            assert para_tensor_gathered.shape[0] == len(moe_raw.experts)
            for expertID in range(para_tensor_gathered.shape[0]):
                list(moe_raw.experts[expertID].parameters())[
                    idx
                ].data = para_tensor_gathered[expertID]

    moe_out, raw_out, moe_grad_in, raw_grad_in = _perform_forward(
        moe, moe_raw, batch_size, d_model, top_k, rank, mp_group
    )

    def get_experts_grad(experts: List[nn.Module]):
        return torch.stack(
            [
                torch.stack(
                    [
                        p.grad.sum() if p.grad is not None else torch.zeros(1).cuda()
                        for p in item.parameters()
                    ]
                ).sum()
                for item in experts
            ]
        )

    moe_grad, raw_grad = (
        get_experts_grad(moe.experts),
        get_experts_grad(moe_raw.experts),
    )

    if world_size > 1:
        torch.distributed.all_reduce(raw_grad)
        mp_size = mp_group.size() if mp_group else 1
        raw_grad = raw_grad[rank * num_expert : (rank + 1) * num_expert] / mp_size

    moe_out_list = [moe_out, moe_grad, moe_grad_in]
    raw_out_list = [raw_out, raw_grad, raw_grad_in]
    names = ["forward", "backward", "grad_in"]

    _assert_numerical(names, moe_out_list, raw_out_list, rank)


class MyModule(nn.Module):
    def __init__(self, dim=8):
        super(MyModule, self).__init__()
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(dim, dim)),
                    ("relu1", nn.ReLU()),
                    ("linear2", nn.Linear(dim, dim)),
                    ("relu2", nn.ReLU()),
                    ("linear3", nn.Linear(dim, dim)),
                ]
            )
        )

    def set_comm(self):
        for p in self.model._modules["linear1"].parameters():
            setattr(p, "dp_comm", "mp")
        for p in self.model._modules["linear2"].parameters():
            setattr(p, "dp_comm", "dp")
        for p in self.model._modules["linear3"].parameters():
            setattr(p, "dp_comm", "world")

    def forward(self, inp):
        return self.model(inp)


def _test_fmoe_local_ddp(rank, world_size, mp_group, dp_group, world_group):
    batch_size, dim = 4, 8

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    model = MyModule().cuda()
    model_ddp = LocalDDP(deepcopy(model),
            mp_group=mp_group, dp_group=dp_group, world_group=world_group)
    model.set_comm()
    model_ddp.module.set_comm()

    inp = torch.randn(batch_size, dim).cuda()

    raw_out = model(inp).mean()
    ddp_out = model_ddp(inp).mean()

    raw_out.backward()
    ddp_out.backward()

    torch.distributed.all_reduce(
        model.model._modules["linear1"].weight.grad.data, group=mp_group
    )
    model.model._modules["linear1"].weight.grad /= mp_group.size()
    torch.distributed.all_reduce(
        model.model._modules["linear2"].weight.grad.data, group=dp_group
    )
    model.model._modules["linear2"].weight.grad /= dp_group.size()
    torch.distributed.all_reduce(
        model.model._modules["linear3"].weight.grad.data, group=world_group
    )
    model.model._modules["linear3"].weight.grad /= world_group.size()
    model_ddp.allreduce_params(reduce_after=False, fp32_allreduce=False)

    raw_out_list = [
        model.model._modules["linear1"].weight.grad,
        model.model._modules["linear2"].weight.grad,
        model.model._modules["linear3"].weight.grad,
    ]
    ddp_out_list = [
        model_ddp.module.model._modules["linear1"].weight.grad,
        model_ddp.module.model._modules["linear2"].weight.grad,
        model_ddp.module.model._modules["linear3"].weight.grad,
    ]

    names = ["mp grad", "dp grad", "wp grad"]

    _assert_numerical(names, ddp_out_list, raw_out_list, rank)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_expert", [None])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("expert", [ [NaiveExpert for _ in range(4)], [LinearExpert, NaiveExpert, LinearExpert, NaiveExpert, LinearExpert, NaiveExpert, LinearExpert, NaiveExpert] ])
@pytest.mark.parametrize("rank", [0])
@pytest.mark.parametrize("world_size", [1])
@pytest.mark.parametrize("mp_group", [None])
@pytest.mark.parametrize("dp_group", [None])
@pytest.mark.parametrize("world_group", [None])
def test_fmoe_experts(
    batch_size,
    num_expert,
    d_model,
    top_k,
    expert: Union[Type[nn.Module], str],
    rank,
    world_size,
    mp_group,
    dp_group,
    world_group,
):
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    if isinstance(expert, str):
        expert = globals()[expert]

    moe = FMoE(
        num_expert=num_expert,
        d_model=d_model,
        gate=NaiveGate,
        world_size=world_size,
        mp_group=mp_group,
        expert=expert,
        top_k=top_k,
    ).cuda()

    moe_raw = BruteForceMoE(
        expert=expert,
        num_expert=num_expert,
        d_model=d_model,
        world_size=world_size,
        top_k=top_k,
    ).cuda()

    if world_size == 1:
        for expert_moe, expert_raw in zip(moe.experts, moe_raw.experts):
            for para_moe, para_raw in zip(
                expert_moe.parameters(), expert_raw.parameters()
            ):
                para_raw.data = para_moe.data.clone()
    else:
        assert len(moe.experts) >= 1
        for idx, para in enumerate(moe.experts[0].parameters()):
            para_tensor = torch.cat(
                [list(expert.parameters())[idx].unsqueeze(0) for expert in moe.experts]
            )
            para_array = [torch.empty_like(para_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(para_array, para_tensor)
            para_tensor_gathered = torch.cat(para_array, dim=0)
            assert para_tensor_gathered.shape[0] == len(moe_raw.experts)
            for expertID in range(para_tensor_gathered.shape[0]):
                list(moe_raw.experts[expertID].parameters())[
                    idx
                ].data = para_tensor_gathered[expertID]

    moe_out, raw_out, moe_grad_in, raw_grad_in = _perform_forward(
        moe, moe_raw, batch_size, d_model, top_k, rank, mp_group
    )

    def get_experts_grad(experts: List[nn.Module]):
        return torch.stack(
            [
                torch.stack(
                    [
                        p.grad.sum() if p.grad is not None else torch.zeros(1).cuda()
                        for p in item.parameters()
                    ]
                ).sum()
                for item in experts
            ]
        )

    moe_grad, raw_grad = (
        get_experts_grad(moe.experts),
        get_experts_grad(moe_raw.experts),
    )

    if world_size > 1:
        torch.distributed.all_reduce(raw_grad)
        mp_size = mp_group.size() if mp_group else 1
        raw_grad = raw_grad[rank * num_expert : (rank + 1) * num_expert] / mp_size

    moe_out_list = [moe_out, moe_grad, moe_grad_in]
    raw_out_list = [raw_out, raw_grad, raw_grad_in]
    names = ["forward", "backward", "grad_in"]

    _assert_numerical(names, moe_out_list, raw_out_list, rank)


if __name__ == "__main__":
    test_fmoe_linear(
        batch_size=2,
        num_expert=2,
        d_model=2,
        top_k=2,
        d_hidden=16,
        rank=0,
        world_size=1,
        mp_group=None,
        dp_group=None,
        world_group=None,
        data_type=torch.float32,
    )
