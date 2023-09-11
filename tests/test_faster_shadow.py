import pytest

import os
import sys
import json
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fmoe.functions import ensure_comm
from test_ddp import _ensure_initialized, _run_distributed
from test_numerical import _assert_numerical
from fmoe.fastermoe.schedule import _fmoe_general_global_forward as smart_fwd
from fmoe.layers import _fmoe_general_global_forward as naive_fwd


@pytest.mark.parametrize("n_process", [8])
@pytest.mark.parametrize("d_model", [1024])
@pytest.mark.parametrize("batch_size", [16, 512])
@pytest.mark.parametrize("n_expert", [1])
@pytest.mark.parametrize("group_sz", [1, 2, 4])
@pytest.mark.parametrize("pass_stored", [True, False])
def test_faster_shadow(n_process, d_model, batch_size, n_expert, group_sz, pass_stored):
    _run_distributed('_test_faster_shadow',
            n_process,
            {
                'd_model': d_model,
                'batch_size': batch_size,
                'n_expert': n_expert,
                'pass_stored': pass_stored
            },
            script=__file__,
            env=dict(
                FMOE_FASTER_GROUP_SIZE=str(group_sz),
                FMOE_FASTER_SHADOW_ENABLE='ON'
            )
    )


def _test_faster_shadow(d_model, batch_size, n_expert, pass_stored):
    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    x1 = torch.rand(batch_size, d_model).cuda()
    x1.requires_grad = True
    x2 = x1.data.clone()
    x2.requires_grad = True
    topk_idx = torch.randint(0, world_size * n_expert, (batch_size, 2)).cuda()
    m1 = torch.nn.Linear(d_model, d_model).cuda()
    m2 = torch.nn.Linear(d_model, d_model).cuda()
    with torch.no_grad():
        m2.weight.copy_(m1.weight)
        m2.bias.copy_(m1.bias)

    def ef1(x, fec, eidx):
        y = m1(x)
        return y
    def ef2(x, fec):
        y = m2(x)
        return y

    if pass_stored:
        stored_models = torch.randint(0, 2, (world_size * n_expert,)).bool().cuda()
        while stored_models.sum().item() == 0:
            stored_models = torch.randint(0, 2, (world_size * n_expert,)).bool().cuda()
        stored_models[-1] = True
        dist.broadcast(stored_models, 0)
        stored_models = stored_models.cpu()
        print(stored_models)

    ensure_comm(x1, None)
    if pass_stored:
        y1 = smart_fwd(x1, topk_idx, ef1, n_expert, world_size, experts=[m1],
                stored_models=stored_models)
    else:
        y1 = smart_fwd(x1, topk_idx, ef1, n_expert, world_size, experts=[m1])
    y1.sum().backward()

    y2 = naive_fwd(x2, topk_idx, ef2, n_expert, world_size, experts=[m2])
    y2.sum().backward()
    _assert_numerical(['out', 'grad_in', 'grad_bias', 'grad_weight'],
            [y1, x1.grad, m1.bias.grad, m1.weight.grad],
            [y2, x2.grad, m2.bias.grad, m2.weight.grad], rank)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        locals()[sys.argv[1]](**args)
    else:
        # test_faster_shadow(8, 16, 16, 1, 2)
        _test_faster_shadow(1024, 16, 1, True)
