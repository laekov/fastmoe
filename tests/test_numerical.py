from moe import FMoE as MOELayer 
from moe import BruteForceMoE as MOELayer_raw
import torch
from torch import nn
import sys
import os


rank = None
world_size = None


def test_moe():
    def test_module(moe, linear, inp, gate):
        linear.zero_grad()
        moe.zero_grad()
        x = (linear(inp))
        output = moe(x, gate)
        y = output.mean()
        y.backward()
        return output, moe.weight.grad, linear.weight.grad, linear.bias.grad

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    batch_size = 4
    num_expert = 2
    in_feat = 6
    out_feat = 7

    linear = nn.Linear(in_feat, in_feat).cuda()

    moe = MOELayer(num_expert, in_feat, out_feat, world_size).cuda()
    moe_raw = MOELayer_raw(num_expert, in_feat, out_feat, world_size).cuda()
    if world_size == 1:
        moe_raw.weight.data = moe.weight.data.clone()
    else:
        weight_array = [torch.empty_like(moe.weight.data)
                for _ in range(world_size)]
        torch.distributed.all_gather(weight_array, moe.weight.data)
        moe_raw.weight.data = torch.cat(weight_array, dim=0)

    inp = torch.rand(batch_size, in_feat).cuda()
    gate = torch.randint(low=0, 
            high=num_expert * world_size, 
            size=(batch_size,), 
            requires_grad=False).int().cuda()
    # gate = torch.Tensor([0, 1, 0, 1]).int().cuda()

    moe_out = test_module(moe, linear, inp.clone(), gate.clone())
    raw_out = test_module(moe_raw, linear, inp.clone(), gate.clone())

    names = ['Out', 'Moe wei', 'Linear wei', 'Linear bias']
    if world_size > 1:
        ou, wg, lwg, lbg = raw_out
        torch.distributed.all_reduce(wg)
        wg = wg[rank * num_expert:(rank + 1)* num_expert]
        raw_out = ou, wg, lwg, lbg
    for name, mo, ro in zip(names, moe_out, raw_out):
        err = (mo - ro).abs().sum()
        print('Rank {} {} abs err {}'.format(rank, name, err))
        if err > 1e-3:
            sys.stderr.write('=========== moe out ==============\n')
            sys.stderr.write('{}\n'.format(mo)) 
            sys.stderr.write('=========== raw out ==============\n')
            sys.stderr.write('{}\n'.format(ro)) 
            return


if __name__ == '__main__':
    os.environ['RANK'] = os.environ.get('OMPI_COMM_WORLD_RANK', '0')
    os.environ['WORLD_SIZE'] = os.environ.get('OMPI_COMM_WORLD_SIZE', '1')
    if int(os.environ['WORLD_SIZE']) > 1:
        torch.distributed.init_process_group(backend='nccl')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    test_moe()
