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


@pytest.mark.parametrize("n", [1, 2])
def test_ensure(n):
    _run_distributed('_test_ensure',
            n, dict(),
            script=__file__
    )


def _test_ensure():
    _ensure_initialized()
    rank = torch.distributed.get_rank()
    x = torch.rand(10).cuda()
    ensure_comm(x, None)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        locals()[sys.argv[1]](**args)
    else:
        _ensure_initialized()
        _test_ensure()
