import pytest

import os
import sys
import json
import math

import torch
import torch.distributed as dist
from fmoe.gates import GShardGate, SwitchGate
from test_ddp import _run_distributed


def _ensure_initialized():
    if not dist.is_initialized():
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12211")
        dist.init_process_group(backend="nccl")


@pytest.mark.parametrize("d_model", [1024])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("n_expert", [1, 4])
@pytest.mark.parametrize("cap", [.1, 1.1])
def test_gshard_gate(d_model, batch_size, n_expert, cap):
    if 1 * n_expert < 2:
        pytest.skip("No enough experts")
    _run_distributed('_test_gshard_gate',
            1,
            {
                'd_model': d_model,
                'batch_size': batch_size,
                'n_expert': n_expert,
                'cap': cap
            },
            script=__file__
    )


def _test_gshard_gate(d_model, batch_size, n_expert, cap):
    _ensure_initialized()
    gate = GShardGate(d_model, n_expert, dist.get_world_size(),
            capacity=(cap, cap)).cuda()
    x = torch.rand(batch_size, d_model).cuda()
    topk_idx, topk_val = gate(x)
    counts = [0 for _ in range(n_expert * dist.get_world_size())]
    for v in topk_idx.cpu().view(-1).numpy():
        if v != -1:
            counts[v] += 1
    real_cap = math.ceil(cap * batch_size)
    for i in counts:
        assert(i <= real_cap)


@pytest.mark.parametrize("d_model", [1024])
@pytest.mark.parametrize("batch_size", [4096])
@pytest.mark.parametrize("n_expert", [1, 16])
@pytest.mark.parametrize("cap", [.1, .8])
def test_switch_gate(d_model, batch_size, n_expert, cap):
    _run_distributed('_test_switch_gate',
            1,
            {
                'd_model': d_model,
                'batch_size': batch_size,
                'n_expert': n_expert,
                'cap': cap
            },
            script=__file__
    )


def _test_switch_gate(d_model, batch_size, n_expert, cap):
    _ensure_initialized()
    gate = SwitchGate(d_model, n_expert, dist.get_world_size(),
            capacity=(cap, cap)).cuda()
    x = torch.rand(batch_size, d_model).cuda()
    topk_idx, topk_val = gate(x)
    counts = [0 for _ in range(n_expert * dist.get_world_size())]
    for v in topk_idx.cpu().view(-1).numpy():
        if v != -1:
            counts[v] += 1
    real_cap = math.ceil(cap * batch_size)
    for i in counts:
        assert(i <= real_cap)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        locals()[sys.argv[1]](**args)
    else:
        _ensure_initialized()
        # test_gshard_gate(4096, 1024, 4, .2)
        test_gshard_gate(8, 16, 1, .1)
        # test_switch_gate(4096, 1024, 4, .2)
