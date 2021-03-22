r"""
Different implementations of the Gate are located here.
The `NaiveGate` is the reference to implement any other gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroGate(nn.Module):
    r"""
    Guide all input samples to gate 0.
    """

    def __init__(self, _1, _2, _3, top_k=2):
        super().__init__()
        self.top_k = top_k

    def forward(self, inp):
        r"""
        All output to expert 1
        """
        idx = torch.zeros(
            inp.shape[0] * self.top_k, dtype=torch.int64, device=inp.device
        )
        score = torch.ones(inp.shape[0] * self.top_k,
                device=inp.device) / self.top_k
        return idx, score.reshape(-1, 1, self.top_k)


class NaiveGate(nn.Module):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indecies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__()
        self.gate = nn.Linear(d_model, num_expert * world_size)
        self.top_k = top_k

    def forward(self, inp):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1).unsqueeze(1)
        gate_top_k_idx = gate_top_k_idx.view(-1)  # (BxLxtop_k)

        return gate_top_k_idx, gate_score
