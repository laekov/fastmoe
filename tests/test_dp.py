from moe import FMoE as MOELayer 
from moe import BruteForceMoE as MOELayer_raw
import torch
from torch import nn
import sys
import os


n_devices = int(os.environ.get('N_GPUS', '2'))


def test_dp():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    batch_size = 6
    num_expert = 4
    in_feat = 2
    out_feat = 3

    inp = torch.rand(batch_size, in_feat).cuda()
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), 
            requires_grad=False).cuda()

    print("data parallel of our MoE model")
    moe = MOELayer(num_expert, in_feat, out_feat).cuda()
    moe_dp = torch.nn.DataParallel(moe, device_ids=list(range(n_devices)))
    for i in range(5):
        output = moe_dp(inp, gate)
    print('Successful')


if __name__ == '__main__':
    test_dp()
