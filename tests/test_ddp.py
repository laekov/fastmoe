import json
import random
import os
import sys
from typing import Dict
import random
import socket as sock

import pytest
import torch
import torch.distributed as dist

from test_numerical import test_fmoe as _test_fmoe
from test_numerical import test_fmoe_linear as _test_fmoe_linear
from test_numerical import _test_fmoe_local_ddp


def _ensure_initialized():
    if 'RANK' not in os.environ:
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


port_count = 0

def _run_distributed(func, world_size, args: Dict, script=__file__, env=dict()):
    device_count = torch.cuda.device_count()
    if device_count < world_size:
        pytest.skip("No enough GPU, only {} found".format(device_count))
    import subprocess
    import os

    ps = []
    env["MASTER_ADDR"] = "localhost"
    global port_count
    env["MASTER_PORT"] = str(9010 + port_count)
    port_count += 1
    env["OMPI_COMM_WORLD_SIZE"] = str(world_size)
    env["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")

    for i in range(world_size):
        env["OMPI_COMM_WORLD_RANK"] = str(i)
        p = subprocess.Popen(
            [sys.executable, script, func, json.dumps(args)],
            stdout=subprocess.PIPE,
            env=env
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
@pytest.mark.parametrize("data_type", ['torch.float32', 'torch.bfloat16', 'torch.float16'])
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
@pytest.mark.parametrize("data_type", ['torch.float32', 'torch.bfloat16', 'torch.float16'])
def test_fmoe_distributed(num_expert, top_k, batch_size, d_model, expert, mp_size, data_type):
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
            "data_type": data_type,
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
        torch.distributed.init_process_group(backend="nccl")
        args = dict(mp_size=1, data_type='torch.float16')
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
        _test_fmoe(4, 2, 16, 2, 'NaiveExpert', **args)
