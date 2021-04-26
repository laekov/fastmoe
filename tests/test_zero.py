import torch
from fmoe.layers import _fmoe_general_global_forward
from fmoe import FMoETransformerMLP


class ConstantGate(torch.nn.Module):
    def __init__(self, d_model, num_expert, world_size, top_k=1):
        super().__init__()
        self.top_k = top_k

    def forward(self, inp):
        idx = torch.zeros((inp.shape[0] * self.top_k,), dtype=torch.int64,
                device=inp.device)
        score = torch.ones((inp.shape[0], 1, self.top_k), device=inp.device) / 2
        return idx, score


def test_zero_fwd(num_expert=2, batch_size=4, d_hidden=8, world_size=1):
    inp = torch.rand(batch_size, d_hidden).cuda()
    gate = torch.zeros(batch_size, dtype=torch.int64).cuda()
    x = _fmoe_general_global_forward(inp, gate, lambda x, y: x, num_expert,
            world_size)


def test_zero_transformer(num_expert=2, batch_size=4, d_hidden=8, world_size=1):
    inp = torch.rand(batch_size, d_hidden).cuda()
    model = FMoETransformerMLP(num_expert, d_hidden, d_hidden * 4, world_size,
            gate=ConstantGate).cuda()
    oup = model(inp)


if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
    # test_zero_fwd(world_size=torch.distributed.get_world_size())
    test_zero_transformer(num_expert=16, batch_size=4096, d_hidden=1024,
            world_size=torch.distributed.get_world_size())
    print('done')
