from moe import FMoE as MOELayer 
from moe import BruteForceMoE as MOELayer_raw
import torch
from torch import nn
import time
import sys
import os


dev_name_default = 'cuda:0'


def perf():
    torch.manual_seed(42 + torch.distributed.get_rank())
    torch.cuda.manual_seed(42 + torch.distributed.get_rank())
    
    if len(sys.argv) == 6:
        batch_size = int(sys.argv[2])
        in_feat = int(sys.argv[3])
        out_feat = int(sys.argv[4])
        num_expert = int(sys.argv[5])
    else:
        batch_size = 4096
        in_feat = 1024
        out_feat = 4096
        num_expert = 4
    if torch.distributed.get_rank() == 0:
        print('Performance test case bs {} {}x{} ne {}'.format(batch_size,
            in_feat, out_feat, num_expert))
    if torch.distributed.get_world_size() > 1:
        dev_name = 'cuda'
    else:
        dev_name = dev_name_default

    inp = torch.rand(batch_size, in_feat).cuda(dev_name)
    gate = torch.randint(low=0, 
            high=num_expert * torch.distributed.get_world_size(), 
            size=(batch_size, ), requires_grad=False).int().cuda(dev_name)

    moe = MOELayer(num_expert, in_feat, out_feat, world_size).cuda(dev_name)
    moe.train()

    o = moe(inp, gate)

    o = moe(inp, gate)
    o = moe(inp, gate)
    o = moe(inp, gate)

    n_runs = 16
    tott = 0.
    backt = 0.
    maxt = 0.
    sqtot = 0.
    for i in range(n_runs):
        gate = torch.randint(low=0, 
                high=num_expert * torch.distributed.get_world_size(), 
                size=(batch_size, ), requires_grad=False).int().cuda(dev_name)
        ts = time.time()
        o = moe(inp, gate)
        te = time.time()

        loss = o.sum()

        bts = time.time()
        loss.backward()
        bte = time.time()

        tott += te - ts
        sqtot += (te - ts)**2
        maxt = max(maxt, te - ts)
        backt = bte - bts

    gflops = 2e-9 * n_runs * in_feat * out_feat * batch_size / tott
    print('Time mean/max/stdev/back {:.3f} {:.3f} {:.3f} {:.3f} ms, {:.3f} GFLOPs'.format(
        tott * 1e3 / n_runs, maxt * 1e3, 
        (sqtot / n_runs - (tott / n_runs)**2) * 1e3 / n_runs, 
        backt * 1e3 / n_runs, gflops))


def test_module(moe, linear, inp, gate):
    linear.zero_grad()
    moe.zero_grad()
    x = (linear(inp))
    output = moe(x, gate)
    y = output.mean()
    y.backward()
    return output, moe.weight.grad, linear.weight.grad, linear.bias.grad


rank = None
world_size = None


def test():
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


def test_dp():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    batch_size = 6
    num_expert = 4
    in_feat = 2
    out_feat = 3

    inp = torch.rand(batch_size, in_feat).cuda()
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda()

    print("data parallel of a nn.Linear model")
    linear = nn.Linear(in_feat, in_feat).cuda()
    linear_dp = torch.nn.DataParallel(linear, device_ids=[0,1,2])
    output = linear_dp(inp)
    print("successful!")

    print("data parallel of our MoE model")
    moe = MOELayer(num_expert, in_feat, out_feat).cuda()
    moe_dp = torch.nn.DataParallel(moe, device_ids=[0,1,2])
    for i in range(5):
        output = moe_dp(inp, gate)


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
    if len(sys.argv) >= 2:
        task = sys.argv[1]
        print('Specificed task {}'.format(task))
        if task == 'correctness':
            test()
        elif task == 'dp':
            test_dp()
        elif task == 'performance':
            perf()
    else:
        test()
