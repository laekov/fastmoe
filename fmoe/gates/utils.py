r"""
Utilities that may be used in the gates
"""
import torch
from fmoe.functions import count_by_gate
import fmoe_cuda as fmoe_native


def limit_by_capacity(topk_idx, num_expert, world_size, capacity):
    with torch.no_grad():
        capacity = torch.ones(num_expert, dtype=torch.int32,
                device=topk_idx.device) * capacity

        pos, lec, gec = count_by_gate(topk_idx, num_expert, world_size,
                require_pos=False)
        new_gec = fmoe_native.limit_by_capacity(gec, capacity,
                num_expert, world_size)
        if world_size > 1:
            new_lec = fmoe_native.expert_exchange(new_gec, num_expert,
                    world_size)
        else:
            new_lec = new_gec

        topk_idx = fmoe_native.prune_gate_by_capacity(topk_idx,
                new_lec.to(torch.int32), num_expert, world_size)
    return new_lec, new_gec, topk_idx
