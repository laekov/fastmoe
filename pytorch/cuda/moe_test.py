from moe import MOELayer
import torch
import time
import sys


def perf():
    batch_size = int(sys.argv[1])
    in_feat = int(sys.argv[2])
    out_feat = int(sys.argv[3])
    num_expert = int(sys.argv[4])

    inp = torch.rand(batch_size, in_feat).cuda("cuda:1")
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda("cuda:1")

    moe = MOELayer(num_expert, in_feat, out_feat).cuda("cuda:1")

    o = moe(inp, gate)

    n_runs = 16
    tott = 0.
    for i in range(n_runs):
        gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda("cuda:1")
        ts = time.time()
        o = moe(inp, gate)
        te = time.time()
        tott += te - ts

    gflops = 2e-9 * n_runs * in_feat * out_feat * batch_size / tott
    print('Mean time {:.3f} ms, {:.3f} GFLOPs'.format(tott * 1e3 / n_runs, gflops))


if __name__ == '__main__':
    perf()
