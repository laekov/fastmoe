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
    requires_moe_group = True

    def __init__(self, d_model, num_expert, world_size, topk=2):
        super().__init__(d_model, num_expert, world_size, top_k)

    def swipe_once(self, idx, capacity):
        with torch.no_grad():
            idx_new, capacity = fmoe_native.swipe_once(idx, capacity,
                    self.num_expert, self.world_size)
            idx_new = idx_new.to(idx.device)
        return idx_new, capacity


    def forward(self, inp):
        score = self.gate(inp)
        _, orig_idx = torch.topk(gate_score, k=self.top_k, dim=-1)

        if not self.training:
            topk_val = F.softmax(topk_val, dim=-1)
            return topk_idx, topk_val

        capacity = torch.scalar_tensor(inp.shape[0] * self.top_k,
                dtype=torch.long)
        
        topk_idxs = []
        for k in range(self.top_k):
            idx, capacity = self.swipe_once(orig_idx[:, k], capacity)
            topk_idxs.append(idx)
        topk_idx = torch.stack(topk_idxs).transpose(0, 1)
        topk_val = gate_score[idx_x, topk_idx.view(-1)].view(-1, self.top_k)
        topk_val = F.softmax(topk_val, dim=-1)
        return topk_idx, topk_val
