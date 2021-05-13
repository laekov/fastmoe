import pytest
import os
import math
import torch
import torch.distributed as dist
from fmoe.gates import GShardGate, SwitchGate


def _ensure_initialized():
    if not dist.is_initialized():
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12211")
        dist.init_process_group(backend="nccl")


@pytest.mark.parametrize("d_model", [8, 1024])
@pytest.mark.parametrize("batch_size", [16, 4096])
@pytest.mark.parametrize("n_expert", [1, 4, 16])
@pytest.mark.parametrize("cap", [.1, .5, 1.1])
def test_gshard_gate(d_model, batch_size, n_expert, cap):
    _ensure_initialized()
    if dist.get_world_size() * n_expert < 2:
        pytest.skip("No enough experts")
    gate = GShardGate(d_model, n_expert, dist.get_world_size(),
            capacity=(cap, cap)).cuda()
    x = torch.rand(batch_size, d_model).cuda()
    topk_idx, topk_val = gate(x)
    counts = [0 for _ in range(n_expert)]
    for v in topk_idx.cpu().view(-1).numpy():
        if v != -1:
            counts[v] += 1
    real_cap = math.ceil(cap * batch_size)
    for i in counts:
        assert(i <= real_cap)


@pytest.mark.parametrize("d_model", [8, 1024])
@pytest.mark.parametrize("batch_size", [16, 4096])
@pytest.mark.parametrize("n_expert", [1, 4, 16])
@pytest.mark.parametrize("cap", [.1, .5, 1.1])
def test_switch_gate(d_model, batch_size, n_expert, cap):
    _ensure_initialized()
    gate = SwitchGate(d_model, n_expert, dist.get_world_size(),
            capacity=(cap, cap)).cuda()
    x = torch.rand(batch_size, d_model).cuda()
    topk_idx, topk_val = gate(x)
    counts = [0 for _ in range(n_expert)]
    for v in topk_idx.cpu().view(-1).numpy():
        if v != -1:
            counts[v] += 1
    real_cap = math.ceil(cap * batch_size)
    for i in counts:
        assert(i <= real_cap)


if __name__ == '__main__':
    _ensure_initialized()
    # test_gshard_gate(4096, 1024, 4, .2)
    test_switch_gate(4096, 1024, 4, .2)
