import pytest

import os
import sys
import json
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fmoe.functions import ensure_comm
from fmoe.gates.swipe_gate import SwipeGate 
from test_ddp import _ensure_initialized, _run_distributed



@pytest.mark.parametrize("d_model", [1024])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("n_expert", [1, 4])
@pytest.mark.parametrize("top_k", [2, 4])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_swipe_gate(world_size, d_model, batch_size, n_expert, top_k):
    if world_size * n_expert < 2:
        pytest.skip("No enough experts")
    _run_distributed('_test_swipe_gate',
            world_size,
            {
                'd_model': d_model,
                'batch_size': batch_size,
                'n_expert': n_expert,
                'top_k': top_k
            },
            script=__file__
    )


def _test_swipe_gate(d_model, batch_size, n_expert, top_k):
    _ensure_initialized()
    gate = SwipeGate(d_model, n_expert, dist.get_world_size()).cuda()
    x = torch.rand(batch_size, d_model).cuda()
    ensure_comm(x, None)
    topk_idx, topk_val = gate(x)


@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("n_expert", [1, 4])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_swipe_once(world_size, batch_size, n_expert):
    if world_size * n_expert < 2:
        pytest.skip("No enough experts")
    _run_distributed('_test_swipe_once',
            world_size,
            {
                'batch_size': batch_size,
                'n_expert': n_expert
            },
            script=__file__
    )


def _test_swipe_once(batch_size, n_expert):
    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gate = SwipeGate(4, n_expert, dist.get_world_size()).cuda()
    idx = torch.randint(0, n_expert * world_size, (batch_size,)).cuda()
    capacity = torch.scalar_tensor(batch_size * 2, dtype=torch.long)
    ensure_comm(idx, None)
    new_idx, new_cap = gate.swipe_once(idx, capacity, 0)
    idx = torch.randint(0, n_expert * world_size, (batch_size,)).cuda()
    new_idx, new_cap = gate.swipe_once(idx, new_cap, 0)

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        locals()[sys.argv[1]](**args)
    else:
        test_swipe_gate(8, 4, 8, 4, 2)
        # test_swipe_once(8, 800, 4)
