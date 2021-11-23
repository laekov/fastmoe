import sys

import pytest
import torch
import torch.nn as nn
import numpy as np

from fmoe.gates import NaiveGate
from fmoe.layers import FMoE
from fmoe.linear import FMoELinear
from fmoe.megatron.layers import _megatron_init_method


def _assert_numerical(names, moe_out_list, raw_out_list, rank, precision=1e-3):
    for name, mo, ro in zip(names, moe_out_list, raw_out_list):
        err = (mo - ro).abs().max()
        print("Rank {} {} abs err {}".format(rank, name, err))
        if err > precision:
            sys.stderr.write(f"=========== {name} moe out ==============\n")
            sys.stderr.write("{}\n".format(mo))
            sys.stderr.write(f"=========== {name} raw out ==============\n")
            sys.stderr.write("{}\n".format(ro))
            sys.stderr.write(f"=========== {name} diff ==============\n")
            sys.stderr.write("{}\n{}\n".format((mo - ro).abs(), err))
            assert False


class MyExpert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        if type(inp) == dict:
            x = inp["x"]
            y = inp["y"]
        elif type(inp) == list:
            x = inp[0]
            y = inp[1]
        else:
            raise NotImplementedError
        x = self.htoh4(x, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        y = self.htoh4(y, fwd_expert_count)
        y = self.activation(y)
        y = self.h4toh(y, fwd_expert_count)
        if type(inp) == dict:
            ret = {"x": x, "y": y}
        elif type(inp) == list:
            ret = [x, y]

        return ret


class MyGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)

    def forward(self, inp, return_all_scores=False):
        if type(inp) == dict:
            x = inp["x"]
        elif type(inp) == list:
            x = inp[0]
        else:
            raise NotImplementedError
        return super().forward(x, return_all_scores)


class MyMoE(FMoE):
    def __init__(
        self, num_expert, d_model, d_hidden, world_size, mp_group, top_k, activation
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            gate=MyGate,
            world_size=world_size,
            mp_group=mp_group,
            top_k=top_k,
        )
        self.experts = MyExpert(num_expert, d_model, d_hidden, activation)

        rng = np.random.default_rng(1234)
        _megatron_init_method(self.experts.htoh4, rng, 1.0)
        _megatron_init_method(self.experts.h4toh, rng, 1.0)


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("d_hidden", [32])
@pytest.mark.parametrize("rank", [0])
@pytest.mark.parametrize("world_size", [1])
@pytest.mark.parametrize("mp_group", [None])
@pytest.mark.parametrize("dp_group", [None])
@pytest.mark.parametrize("world_group", [None])
@pytest.mark.parametrize(
    "data_type", ["torch.FloatTensor", "torch.DoubleTensor", "torch.HalfTensor"]
)
@pytest.mark.parametrize("list_input", [False, True])
def test_fmoe_mimo_linear(
    num_expert,
    top_k,
    batch_size,
    d_model,
    d_hidden,
    rank,
    world_size,
    mp_group,
    dp_group,
    world_group,
    data_type,
    list_input,
    activation=torch.nn.functional.gelu,
):

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    moe = MyMoE(
        num_expert=num_expert,
        d_model=d_model,
        d_hidden=4 * d_model,
        world_size=world_size,
        mp_group=mp_group,
        top_k=top_k,
        activation=activation,
    ).cuda()

    x = torch.rand(batch_size, d_model).cuda()
    inp = [x, x.clone()] if list_input else {"x": x, "y": x.clone()}
    moe_out = moe(inp)

    if list_input:
        _assert_numerical(["x"], [moe_out[0]], [moe_out[1]], rank)
    else:
        _assert_numerical(["x"], [moe_out["x"]], [moe_out["y"]], rank)


if __name__ == "__main__":
    test_fmoe_mimo_linear(
        batch_size=2,
        num_expert=2,
        d_model=2,
        top_k=2,
        d_hidden=16,
        rank=0,
        world_size=1,
        mp_group=None,
        dp_group=None,
        world_group=None,
        data_type=torch.float32,
        list_input=True
    )
