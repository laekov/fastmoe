#!/usr/bin/env python
# encoding: utf-8
# File Name: topk.py
# Author: Jiezhong Qiu
# Create Time: 2020/11/24 20:23
# TODO:

import torch
import time
from mem_transformer import my_topk
output = torch.rand(16, 512, 512).cuda()

torch.cuda.synchronize()
start = time.time()
_, pred = output.topk(k=1, dim=-1, largest=True, sorted=True)
torch.cuda.synchronize()
print("torch.top1 Time :{}".format(time.time() - start))

torch.cuda.synchronize()
start = time.time()
_, pred_ = my_topk(output, k=1, inplace=False)
torch.cuda.synchronize()
print("my top1 Time :{}".format(time.time() - start))

torch.cuda.synchronize()
start = time.time()
_, pred = output.topk(k=2, dim=-1, largest=True, sorted=True)
torch.cuda.synchronize()
print("torch.top2 Time :{}".format(time.time() - start))

torch.cuda.synchronize()
start = time.time()
_, pred_ = my_topk(output, k=2, inplace=False)
torch.cuda.synchronize()
print("my top2 Time :{}".format(time.time() - start))

