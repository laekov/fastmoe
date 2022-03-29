import pytest

import os
import sys
import json
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from fmoe.gates.faster_gate import FasterGate
from test_ddp import _ensure_initialized, _run_distributed


@pytest.mark.parametrize("n_process", [8])
@pytest.mark.parametrize("d_model", [1024])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("n_expert", [1, 4])
@pytest.mark.parametrize("gpu_per_node", [2, 4, 8])
@pytest.mark.parametrize("frac", [.2])
def test_faster_gate(n_process, d_model, batch_size, n_expert, gpu_per_node, frac):
    _run_distributed('_test_faster_gate',
            n_process,
            {
                'd_model': d_model,
                'batch_size': batch_size,
                'n_expert': n_expert,
                'gpu_per_node': gpu_per_node,
                'frac': frac
            },
            script=__file__,
            env=dict(
                FMOE_TOPO_GPUS_PER_NODE=str(gpu_per_node),
                FMOE_TOPO_OUTGOING_FRACTION=str(frac)
            )
    )


def _test_faster_gate(d_model, batch_size, n_expert, gpu_per_node, frac):
    _ensure_initialized()
    rank = dist.get_rank()
    node_rank = rank // gpu_per_node

    gate = FasterGate(d_model, n_expert, dist.get_world_size(), node_rank).cuda()
    x = torch.rand(batch_size, d_model).cuda()
    topk_idx, topk_val = gate(x)

    cnto = 0
    idxs = topk_idx[:, 0].cpu().view(-1).numpy()
    for v in idxs:
        assert(v != -1)
        if v // n_expert // gpu_per_node != rank // gpu_per_node:
            cnto += 1
    assert(cnto <= math.ceil(batch_size * frac))


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        locals()[sys.argv[1]](**args)
    else:
        test_faster_gate(8, 1024, 16, 1, 2, .2)
