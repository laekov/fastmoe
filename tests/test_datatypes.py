import torch
import pytest

from fmoe.layers import FMoE
from fmoe.transformer import _Expert

class MyMoE(FMoE):
        def __init__(self, num_expert, d_model, d_hidden, top_k, activation):
            super().__init__(
                num_expert=num_expert,
                d_model=d_model,
                top_k=top_k,
            )
            self.experts = _Expert(num_expert, d_model, d_hidden, activation)


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("d_hidden", [32])
@pytest.mark.parametrize("data_type", ['torch.FloatTensor', 'torch.DoubleTensor', 'torch.HalfTensor'])
def test_fmoe_data_support(
    num_expert,
    top_k,
    batch_size,
    d_model,
    d_hidden,
    data_type,
    activation=torch.nn.functional.gelu,
):
    """
        The objective of this test is to make sure that the cuda 
        kernels for forward/backward handle different data types
        without crashing
    """
    moe = MyMoE(
        num_expert, d_model, d_hidden, top_k, activation
    ).type(data_type).cuda()

    inp = torch.rand(batch_size, d_model).type(data_type).cuda()

    moe(inp).sum().backward()



    