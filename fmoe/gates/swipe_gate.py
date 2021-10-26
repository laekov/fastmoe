r"""
Balanced gate using SWIPE algorithm
"""
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from .naive_gate import NaiveGate

from fmoe.functions import count_by_gate
import fmoe_cuda as fmoe_native


class SwipeGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(d_model, num_expert, world_size, top_k)

    def swipe_once(self, idx, capacity, bias):
        with torch.no_grad():
            idx_new, capacity = fmoe_native.swipe_once(idx, capacity,
                    self.num_expert, self.world_size, bias)
            idx_new = idx_new.to(idx.device)
        return idx_new, capacity


    def forward(self, inp):
        score = self.gate(inp)
        orig_score, orig_idx = torch.topk(score, k=self.top_k, dim=-1)

        if not self.training:
            topk_val = F.softmax(orig_score, dim=-1)
            return orig_idx, topk_val

        capacity = torch.scalar_tensor(inp.shape[0] * self.top_k,
                dtype=torch.long)

        topk_idxs = []
        topk_vals = []
        idx_x = torch.arange(inp.shape[0], device=inp.device)
        for k in range(self.top_k):
            idx, capacity = self.swipe_once(orig_idx[:, k], capacity,
                    k % self.num_expert)
            topk_vals.append(score[idx_x, idx])
            topk_idxs.append(idx)
        topk_idx = torch.stack(topk_idxs).transpose(0, 1)
        topk_val = torch.stack(topk_vals).transpose(0, 1)
        topk_val = F.softmax(topk_val, dim=-1)
        return topk_idx, topk_val
