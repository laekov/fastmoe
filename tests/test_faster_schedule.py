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
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("n_expert", [1])
def test_faster_schedule(n_process, d_model, batch_size, n_expert):
    _run_distributed('_test_faster_schedule',
            n_process,
            {
                'd_model': d_model,
                'batch_size': batch_size,
                'n_expert': n_expert
            },
            script=__file__,
            env=dict()
    )


def _test_faster_schedule(d_model, batch_size, n_expert):
    _ensure_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = torch.rand(batch_size, d_model).cuda()
    x.requires_grad = True
    topk_idx = torch.randint(0, world_size * n_expert, (batch_size, 2)).cuda()
    m = torch.nn.Linear(d_model, d_model).cuda()

    def expert_fn(x, fec):
        y = m(x)
        return y

    ensure_comm(x, None)
    y = smart_fwd(x, topk_idx, expert_fn, n_expert, world_size)
    z = naive_fwd(x, topk_idx, expert_fn, n_expert, world_size)
    _assert_numerical(['out'], [y], [z], rank)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        locals()[sys.argv[1]](**args)
    else:
        # test_faster_schedule(8, 16, 16, 1)
        _test_faster_schedule(4, 2, 1)
