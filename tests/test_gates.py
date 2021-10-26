import pytest

import os
import sys
import json
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fmoe.gates import GShardGate, SwitchGate
from test_ddp import _ensure_initialized, _run_distributed


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

    gate_score = gate.gate(x)
    for i in range(batch_size):
        for j in range(gate.top_k):
            v = topk_idx[i, j]
            if v != -1:
                assert topk_val[i, j] == gate_score[i, v]


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
    rng = torch.cuda.get_rng_state() # save rng state
    topk_idx, topk_val = gate(x)
    counts = [0 for _ in range(n_expert * dist.get_world_size())]
    for v in topk_idx.cpu().view(-1).numpy():
        if v != -1:
            counts[v] += 1
    real_cap = math.ceil(cap * batch_size)
    for i in counts:
        assert(i <= real_cap)

    score = gate.gate(x)

    if gate.training:
        # reset rng state to make sure noise is the same as in gate.forward()
        torch.cuda.set_rng_state(rng)
        # random uniform number from [1-eps, 1+eps]
        noise = torch.rand_like(score)
        noise = noise * 2 * gate.switch_eps + 1.0 - gate.switch_eps
        score += noise

    # fp32 softmax for numerical stability
    score = F.softmax(score.float(), dim=-1)

    for i in range(batch_size):
        v = topk_idx[i]
        if v != -1:
            assert topk_val[i] == score[i, topk_idx[i]]


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        locals()[sys.argv[1]](**args)
    else:
        _ensure_initialized()
        # test_gshard_gate(4096, 1024, 4, .2)
        test_switch_gate(8, 16, 4, .1)
        # test_switch_gate(4096, 1024, 4, .2)
