import os

import pytest
import torch

from fmoe.gates import NaiveGate
from fmoe.layers import FMoE
from fmoe.transformer import _Expert

n_devices = int(os.environ.get("N_GPUS", "2"))


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("d_hidden", [32])
def test_fmoe_dp(
    num_expert,
    top_k,
    batch_size,
    d_model,
    d_hidden,
    activation=torch.nn.functional.gelu,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    experts = _Expert(num_expert, d_model, d_hidden, activation).cuda()

    def expert_fn(inp, gate):
        return experts(inp, gate)

    moe = FMoE(
        num_expert=num_expert,
        d_model=d_model,
        gate=NaiveGate,
        world_size=1,
        mp_group=None,
        expert_fn=expert_fn,
        top_k=top_k,
    ).cuda()

    moe_dp = torch.nn.DataParallel(moe, device_ids=list(range(n_devices)))

    for i in range(5):
        output = moe_dp(torch.rand(batch_size, d_model).cuda())
