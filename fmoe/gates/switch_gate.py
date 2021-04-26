r"""
Balanced gate with Switch Transformer's policy (Google, 2021)
"""
import torch
import torch.nn.functional as F
from .naive_gate import NaiveGate

class SwitchGate(NaiveGate):
    r"""
    A switch gate implementation
    """

    def __init__(self, d_model, num_expert, world_size,
            switch_eps=.1, capacity=(1.2, 2.4)):
        super().__init__(d_model, num_expert, world_size, top_k=1)
        self.gate = nn.Linear(d_model, num_expert * world_size)
        self.switch_eps = switch_eps
        self.capacity = capacity

    def forward(self, inp):
        r"""
        The switch firstly conduct softmax and then calculates the top-1
        """
        gate = super().forward(inp)
        if self.training:
            # random uniform number from [1-eps, 1+eps]
            noise = torch.rand_like(gate)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            gate += noise

        # fp32 softmax for numerical stability
        gate_score = F.softmax(gate.float(), dim=-1)

        gate_score_top1, gate_idx_top1 = torch.topk(
            gate_score_clip, k=1, dim=-1, largest=True
        )  # [.. x top_k]
        gate_score = gate_score.to(dtype=inp.dtype)
        gate_score_top1 = gate_score_top1.to(dtype=inp.dtype)

        gate_score_top1 = gate_score_top1.unsqueeze(1)
        gate_idx_top1 = gate_idx_top1.view(-1)  # (BxLxtop_k)

        # TODO: capacity limit

        # TODO: not testd, the following code is super dangerous!!!!!!
        gate_updated = gate_idx_top1
        gate_updated = gate_updated[gate_updated > -1]
        fraction_expert = torch.scatter_add(
                torch.zeros(self.tot_expert, device=gate_updated.device),
                0,
                gate_updated,
                torch.ones_like(gate_updated, dtype=torch.float),
            ) / gate_updated.view(-1).size(0)
        prob_expert = gate_score.sum(dim=0) / gate_updated.view(-1).size(0)
        switch_aux_loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.set_loss(switch_aux_loss)
        return gate_idx_top1, gate_score_top1
