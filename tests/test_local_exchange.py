import sys
from collections import OrderedDict
from typing import List, Type, Union

import pytest
import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy
from fmoe.functions import MOEGather, MOEScatter, count_by_gate

from test_numerical import _assert_numerical

@pytest.mark.parametrize("n_expert", [1, 4, 8])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("batch_size", [12])
@pytest.mark.parametrize("d_model", [6])
@pytest.mark.parametrize("world_size", [1])
def test_scatter(n_expert, topk, batch_size, d_model, world_size):
    gate_idx = torch.randint(n_expert + 1, (batch_size, topk)) - 1
    gate_idx = gate_idx.long().cuda()
    pos, lec, gec = count_by_gate(gate_idx, n_expert, world_size)
    fbs = int(gec.sum().item())
    inp = torch.rand(batch_size, d_model).cuda()
    inp.requires_grad = True
    out = MOEScatter.apply(inp, pos % batch_size, lec, gec, fbs, world_size)
    out.sum().backward()

    inp_raw = inp.data.clone()
    out_raw = torch.empty(pos.shape[0], d_model,
            device=inp.device, dtype=inp.dtype)
    # out_raw.sum().backward()
    for i, f in enumerate(pos.cpu()):
        out_raw[i] = inp[f % batch_size]
    _assert_numerical(['out'], [out], [out_raw], 0)
    # TODO: check grad

if __name__ == '__main__':
    test_scatter(4, 2, 8, 6, 1)
