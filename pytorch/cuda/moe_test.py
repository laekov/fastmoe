from moe import MOELayer, MOELayer_raw
import torch
import time
import sys


def perf():
    torch.manual_seed(42 + torch.distributed.get_rank())
    torch.cuda.manual_seed(42 + torch.distributed.get_rank())
    
    batch_size = int(sys.argv[1])
    io_feat = int(sys.argv[2])
    hidden_feat = int(sys.argv[3])
    num_expert = int(sys.argv[4])

    inp = torch.rand(batch_size, io_feat).cuda()
    gate = torch.randint(low=0, 
            high=num_expert * torch.distributed.get_world_size(), 
            size=(batch_size, ), requires_grad=False).int().cuda()

    moe = MOELayer(num_expert, io_feat, hidden_feat, io_feat).cuda()

    o = moe(inp, gate)
    o = moe(inp, gate)
    o = moe(inp, gate)
    o = moe(inp, gate)
    o = moe(inp, gate)
    o = moe(inp, gate)

    n_runs = 16
    tott = 0.
    maxt = 0.
    sqtot = 0.
    for i in range(n_runs):
        gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda()
        ts = time.time()
        o = moe(inp, gate)
        te = time.time()
        tott += te - ts
        sqtot += (te - ts)**2
        maxt = max(maxt, te - ts)

    gflops = 2e-9 * n_runs * io_feat * hidden_feat * 2 * batch_size / tott
    print('Time mean/max/stdev {:.3f} {:.3f} {:.3f} ms, {:.3f} GFLOPs'.format(
        tott * 1e3 / n_runs, maxt * 1e3, 
        (sqtot / n_runs - (tott / n_runs)**2) * 1e3 / n_runs, gflops))


if __name__ == '__main__':
    torch.distributed.init_process_group(backend='mpi')
    # print('{} / {}'.format(torch.distributed.get_rank(), torch.distributed.get_world_size()))
    perf()
