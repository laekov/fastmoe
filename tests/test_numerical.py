import json
import os
import sys
from typing import List, Callable, Dict, Type, Union

import pytest
import torch
import torch.nn as nn

from fmoe.gates import NaiveGate
from fmoe.layers import FMoE
from fmoe.transformer import _Expert
from moe import BruteForceMoELinear, BruteForceMoE, NaiveExpert, LinearExpert

rank = 0
world_size = 1


def _perform_forward(moe: nn.Module, moe_raw: nn.Module, batch_size, d_model, top_k):
    moe.zero_grad()
    moe_raw.zero_grad()
    inp = torch.rand(batch_size, d_model).cuda()
    gate_idx, gate_score = moe.gate(inp)
    inp_repeated = inp.repeat_interleave(repeats=top_k, dim=0)
    moe_out = moe(inp).mean()
    raw_out = moe_raw(inp_repeated, gate_idx, gate_score).mean()

    moe_out.backward()
    raw_out.backward()

    return moe_out, raw_out


def _assert_numercial(names, moe_out_list, raw_out_list):
    for name, mo, ro in zip(names, moe_out_list, raw_out_list):
        err = (mo - ro).abs().sum()
        print("Rank {} {} abs err {}".format(rank, name, err))
        if err > 1e-3:
            sys.stderr.write("=========== moe out ==============\n")
            sys.stderr.write("{}\n".format(mo))
            sys.stderr.write("=========== raw out ==============\n")
            sys.stderr.write("{}\n".format(ro))
            assert False


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("d_hidden", [32])
def test_fmoe_linear(
    num_expert,
    top_k,
    batch_size,
    d_model,
    d_hidden,
    activation=torch.nn.functional.gelu,
):
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    experts = _Expert(num_expert, d_model, d_hidden, activation).cuda()

    def expert_fn(inp, gate):
        return experts(inp, gate)

    moe = FMoE(
        num_expert=num_expert,
        d_model=d_model,
        gate=NaiveGate,
        world_size=world_size,
        mp_group=None,
        expert_fn=expert_fn,
        top_k=top_k,
    ).cuda()

    moe_raw = BruteForceMoELinear(
        activation=activation,
        num_expert=num_expert,
        d_model=d_model,
        d_hidden=d_hidden,
        world_size=world_size,
    ).cuda()

    if world_size == 1:
        moe_raw.weight_htoh4.data = experts.htoh4.weight.data.clone()
        moe_raw.weight_h4toh.data = experts.h4toh.weight.data.clone()
    else:
        weight_htoh4_array = [
            torch.empty_like(experts.htoh4.weight.data) for _ in range(world_size)
        ]
        torch.distributed.all_gather(weight_htoh4_array, experts.htoh4.weight.data)
        moe_raw.weight_htoh4.data = torch.cat(weight_htoh4_array, dim=0)

        weight_h4toh_array = [
            torch.empty_like(experts.h4toh.weight.data) for _ in range(world_size)
        ]
        torch.distributed.all_gather(weight_h4toh_array, experts.h4toh.weight.data)
        moe_raw.weight_h4toh.data = torch.cat(weight_h4toh_array, dim=0)

    moe_out, raw_out = _perform_forward(moe, moe_raw, batch_size, d_model, top_k)

    moe_out_list = moe_out, experts.htoh4.weight.grad, experts.h4toh.weight.grad
    raw_out_list = raw_out, moe_raw.weight_htoh4.grad, moe_raw.weight_h4toh.grad

    if world_size > 1:
        _, htoh4_grad, h4toh_grad = raw_out_list
        torch.distributed.all_reduce(htoh4_grad)
        torch.distributed.all_reduce(h4toh_grad)
        htoh4_grad = htoh4_grad[rank * num_expert : (rank + 1) * num_expert]
        h4toh_grad = h4toh_grad[rank * num_expert : (rank + 1) * num_expert]
        raw_out_list = _, htoh4_grad, h4toh_grad

    names = ["output", "htoh4 weight grad", "h4toh weight grad"]
    _assert_numercial(names, moe_out_list, raw_out_list)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("expert", ["NaiveExpert", "LinearExpert"])
def test_fmoe(
    batch_size, num_expert, d_model, top_k, expert: Union[Type[nn.Module], str]
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
        mp_group=None,
        expert=expert,
        top_k=top_k,
    ).cuda()

    moe_raw = BruteForceMoE(
        expert=expert, num_expert=num_expert, d_model=d_model, world_size=world_size,
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

    moe_out, raw_out = _perform_forward(moe, moe_raw, batch_size, d_model, top_k)

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
        raw_grad = raw_grad[rank * num_expert : (rank + 1) * num_expert]

    moe_out_list = [moe_out, moe_grad]
    raw_out_list = [raw_out, raw_grad]
    names = ["forward", "backward"]

    _assert_numercial(names, moe_out_list, raw_out_list)


def _run_distributed(func: Callable, args: Dict):
    import subprocess
    import os

    ps, n = [], 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "36666"
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(n)

    for i in range(n):
        os.environ["OMPI_COMM_WORLD_RANK"] = str(i)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p = subprocess.Popen(
            [sys.executable, __file__, func.__name__, json.dumps(args)],
            stdout=subprocess.PIPE,
        )
        ps.append(p)

    for p in ps:
        p.wait()
        retc = p.poll()
        assert retc == 0


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("d_hidden", [32])
def test_fmoe_linear_distributed(
    num_expert, top_k, batch_size, d_model, d_hidden,
):
    _run_distributed(
        test_fmoe_linear,
        {
            "num_expert": num_expert,
            "top_k": top_k,
            "batch_size": batch_size,
            "d_model": d_model,
            "d_hidden": d_hidden,
        },
    )


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("expert", ["NaiveExpert", "LinearExpert"])
def test_fmoe_distributed(
    num_expert, top_k, batch_size, d_model, expert,
):
    _run_distributed(
        test_fmoe,
        {
            "num_expert": num_expert,
            "top_k": top_k,
            "batch_size": batch_size,
            "d_model": d_model,
            "expert": expert,
        },
    )


if __name__ == "__main__":
    os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
    if int(os.environ["WORLD_SIZE"]) > 1:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    if len(sys.argv) >= 3:
        locals()[sys.argv[1]](**json.loads(sys.argv[2]))
    else:
        test_fmoe_linear(batch_size=4, num_expert=4, d_model=8, top_k=2, d_hidden=16)
        test_fmoe(batch_size=4, num_expert=4, d_model=8, top_k=2, expert=NaiveExpert)
