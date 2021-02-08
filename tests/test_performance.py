from fmoe import FMoETransformerMLP as MOELayer 
import torch
import time
import sys
import os


rank = None
world_size = None
dev_name_default = 'cuda:0'


def test_performance(batch_size, in_feat, hidden_feat, num_expert, top_k):
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    
    if rank == 0:
        print('Performance test case bs {} {}x{} ne {}x{} topk {}'.format(
            batch_size, in_feat, hidden_feat, world_size, num_expert, top_k))
    if world_size > 1:
        dev_name = 'cuda'
    else:
        dev_name = dev_name_default

    inp = torch.rand(batch_size, in_feat).cuda(dev_name)
    inp.requires_grad = True

    moe = MOELayer(num_expert=num_expert,
            d_model=in_feat, d_hidden=hidden_feat, 
            world_size=world_size, top_k=top_k).cuda(dev_name)
    moe.train()

    # warm up
    for _ in range(4):
        _ = moe(inp)

    n_runs = 16
    tott = 0.
    backt = 0.
    maxt = 0.
    sqtot = 0.
    for i in range(n_runs):
        ts = time.time()
        o = moe(inp)
        te = time.time()

        loss = o.sum()

        bts = time.time()
        loss.backward()
        bte = time.time()

        tott += te - ts
        sqtot += (te - ts)**2
        maxt = max(maxt, te - ts)
        backt = bte - bts

    gflops = 2e-9 * n_runs * (in_feat * hidden_feat * batch_size * top_k * 2 +
            batch_size * in_feat * num_expert) / tott
    print('Time mean/max/stdev/back {:.3f} {:.3f} {:.3f} {:.3f} ms, {:.3f} GFLOPs'.format(
        tott * 1e3 / n_runs, maxt * 1e3, 
        (sqtot / n_runs - (tott / n_runs)**2) * 1e3 * top_k / n_runs, 
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
   
    test_performance(4096, 1024, 4096, 8, 8)
