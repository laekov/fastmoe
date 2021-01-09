from moe import MOELayer
import torch
import time
import sys


def perf():
    batch_size = int(sys.argv[1])
    io_feat = int(sys.argv[2])
    hidden_feat = int(sys.argv[3])
    num_expert = int(sys.argv[4])


    inp = torch.rand(batch_size, in_feat).cuda("cuda:1")
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda("cuda:1")

    moe = MOELayer(num_expert, in_feat, out_feat).cuda("cuda:1")

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
        gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda("cuda:1")
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
    perf()
