r"""
Balanced gate with GShard's policy (Google, 2020)
"""
import torch
import torch.nn.functional as F
from .naive_gate import NaiveGate


class GShardGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, capacity=(1.2, 2.4)):
        super().__init__(d_model, num_expert, world_size, top_k=2)
        self.capacity = capacity

    def forward(self, x):
        topk_idx, topk_val, gate_score = super().forward(x)

        S = gate_score.shape[0]
        top_k = topk_idx.shape[0] // gate_score.shape[0]
        top1_idx = topk_idx.view((-1, top_k))[:, 0]
        c_e = torch.scatter_add(
                torch.zeros(self.num_expert, device=gate_top_1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
                ) / S
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        # TODO: capacity limit

        return topk_idx, topk_val
