import os
import sys
import json
import torch
from fmoe.layers import _fmoe_general_global_forward
from fmoe import FMoETransformerMLP

from test_ddp import _run_distributed


class ConstantGate(torch.nn.Module):
    def __init__(self, d_model, num_expert, world_size, top_k=1):
        super().__init__()
        self.top_k = top_k

    def forward(self, inp):
        idx = torch.zeros((inp.shape[0], self.top_k), dtype=torch.int64,
                device=inp.device)
        score = torch.ones((inp.shape[0], 1, self.top_k), device=inp.device) / 2
        return idx, score


def test_zero_fwd(num_expert=2, batch_size=4, d_hidden=8, world_size=1):
    _run_distributed('_test_zero_fwd',
            1,
            {
                'num_expert': num_expert,
                'batch_size': batch_size,
                'd_hidden': d_hidden
            },
            script=__file__
    )

def _test_zero_fwd(num_expert=2, batch_size=4, d_hidden=8, world_size=1):
    inp = torch.rand(batch_size, d_hidden).cuda()
    gate = torch.zeros(batch_size, dtype=torch.int64).cuda()
    x = _fmoe_general_global_forward(inp, gate, lambda x, y: x, num_expert,
            world_size)


def test_zero_transformer(num_expert=2, batch_size=4, d_hidden=8, world_size=1):
    _run_distributed('_test_zero_transformer',
            1,
            {
                'num_expert': num_expert,
                'batch_size': batch_size,
                'd_hidden': d_hidden
            },
            script=__file__
    )

def _test_zero_transformer(num_expert=2, batch_size=4, d_hidden=8, world_size=1):
    inp = torch.rand(batch_size, d_hidden).cuda()
    mask = torch.zeros(inp.shape[0], dtype=torch.long)
    mask[1] = 1
    mask_dict = {
        1: torch.zeros(d_hidden).cuda()
    }
    model = FMoETransformerMLP(num_expert, d_hidden, d_hidden * 4,
            world_size=world_size, gate=ConstantGate, mask=mask,
            mask_dict=mask_dict).cuda()
    oup = model(inp)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        torch.distributed.init_process_group(backend="nccl")
        args['world_size'] = torch.distributed.get_world_size()
        locals()[sys.argv[1]](**args)
    else:
        # test_zero_fwd(world_size=torch.distributed.get_world_size())
        test_zero_transformer(num_expert=16, batch_size=4096, d_hidden=1024,
                world_size=1)
        print('done')
