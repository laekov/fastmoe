from moe import FMoE as MOELayer 
import torch
import time
import sys
import os


rank = None
world_size = None
dev_name_default = 'cuda:0'


def test_performance(batch_size, in_feat, out_feat, num_expert):
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    
    if rank == 0:
        print('Performance test case bs {} {}x{} ne {}x{}'.format(
            batch_size, in_feat, out_feat, world_size, num_expert))
    if world_size > 1:
        dev_name = 'cuda'
    else:
        dev_name = dev_name_default

    inp = torch.rand(batch_size, in_feat).cuda(dev_name)
    gate = torch.randint(low=0, 
            high=num_expert * world_size,
            size=(batch_size, ), requires_grad=False).int().cuda(dev_name)

    moe = MOELayer(num_expert, in_feat, out_feat, world_size).cuda(dev_name)
    moe.train()

    # warm up
    for _ in range(4):
        _ = moe(inp, gate)

    n_runs = 16
    tott = 0.
    backt = 0.
    maxt = 0.
    sqtot = 0.
    for i in range(n_runs):
        gate = torch.randint(low=0, 
                high=num_expert * world_size,
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
   
    test_performance(4096, 1024, 4096, 8)
