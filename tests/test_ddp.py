import json
import os
import sys
from typing import Dict

import pytest
import torch
import torch.distributed as dist

from test_numerical import test_fmoe as _test_fmoe
from test_numerical import test_fmoe_linear as _test_fmoe_linear
from test_numerical import _test_fmoe_local_ddp


def _ensure_initialized():
    if not dist.is_initialized():
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12211")
        dist.init_process_group(backend="nccl")


def _run_distributed(func, world_size, args: Dict, script=__file__):
    if torch.cuda.device_count() < world_size:
        pytest.skip("No enough GPU")
    import subprocess
    import os

    ps = []
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "36666"
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(world_size)

    for i in range(world_size):
        os.environ["OMPI_COMM_WORLD_RANK"] = str(i)
        p = subprocess.Popen(
            [sys.executable, script, func, json.dumps(args)], stdout=subprocess.PIPE
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
@pytest.mark.parametrize("mp_size", [1, 2])
@pytest.mark.parametrize("data_type", ['torch.FloatTensor', 'torch.DoubleTensor', 'torch.HalfTensor'])
def test_fmoe_linear_distributed(
    num_expert, top_k, batch_size, d_model, d_hidden, mp_size, data_type
):
    _run_distributed(
        "_test_fmoe_linear",
        mp_size * 2,
        {
            "num_expert": num_expert,
            "top_k": top_k,
            "batch_size": batch_size,
            "d_model": d_model,
            "d_hidden": d_hidden,
            "mp_size": mp_size,
            "data_type": data_type
        },
    )


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("expert", ["NaiveExpert", "LinearExpert"])
@pytest.mark.parametrize("mp_size", [1, 2])
def test_fmoe_distributed(num_expert, top_k, batch_size, d_model, expert, mp_size):
    _run_distributed(
        "_test_fmoe",
        mp_size * 2,
        {
            "num_expert": num_expert,
            "top_k": top_k,
            "batch_size": batch_size,
            "d_model": d_model,
            "expert": expert,
            "mp_size": mp_size,
        },
    )


@pytest.mark.parametrize("mp_size", [1, 2])
def test_fmoe_local_ddp(mp_size):
    _run_distributed(
        _test_fmoe_local_ddp.__name__, mp_size * 2, {"mp_size": mp_size},
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        torch.distributed.init_process_group(backend="nccl")
        args["rank"] = torch.distributed.get_rank()
        args["world_size"] = torch.distributed.get_world_size()
        args["mp_group"] = [
            torch.distributed.new_group(
                ranks=[j * args["mp_size"] + i for i in range(args["mp_size"])],
                backend="nccl",
            )
            for j in range(args["world_size"] // args["mp_size"])
        ][args["rank"] // args["mp_size"]]
        args["dp_group"] = [
            torch.distributed.new_group(
                ranks=[
                    i * args["mp_size"] + j
                    for i in range(args["world_size"] // args["mp_size"])
                ],
                backend="nccl",
            )
            for j in range(args["mp_size"])
        ][args["rank"] % args["mp_size"]]
        args["world_group"] = torch.distributed.new_group(
            ranks=list(range(args["world_size"])), backend="nccl",
        )
        del args["mp_size"]
        locals()[sys.argv[1]](**args)
    else:
        test_fmoe_local_ddp(mp_size=2)
        test_fmoe_linear_distributed(
            num_expert=4, top_k=2, batch_size=4, d_model=8, d_hidden=8, mp_size=2,
            data_type="torch.HalfTensor"
        )
