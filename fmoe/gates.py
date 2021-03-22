r"""
Different implementations of the Gate are located here.
The `NaiveGate` is the reference to implement any other gate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class ZeroGate(nn.Module):
    r"""
    Guide all input samples to gate 0.
    """

    def __init__(self, _1, num_expert, _3, top_k=2):
        super().__init__()
        self.num_expert = num_expert
        self.top_k = top_k

    def forward(self, inp):
        r"""
        All output to expert 1
        """
        idx = torch.zeros(
            inp.shape[0] * self.top_k, dtype=torch.int64, device=inp.device
        )
        gate_score = (
            torch.ones(inp.shape[0] * self.top_k, device=inp.device) / self.top_k
        )
        gate_score_all = torch.zeros(inp.shape[0], self.num_expert, device=inp.device)
        gate_score_all[:, 0] = 1
        return idx, gate_score.reshape(-1, 1, self.top_k), gate_score_all


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

        return gate_top_k_idx, gate_score, gate


class NoisyGate(nn.Module):
    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__()
        self.num_expert = num_expert * world_size
        self.w_gate = nn.Parameter(
            torch.zeros(d_model, num_expert * world_size), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(d_model, num_expert * world_size), requires_grad=True
        )
        self.top_k = top_k
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.noise_epsilon = 1e-2

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(
            torch.tensor([0.0], device=clean_values.device),
            torch.tensor([1.0], device=clean_values.device),
        )
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, inp):
        clean_logits = inp @ self.w_gate
        raw_noise_stddev = inp @ self.w_noise
        noise_stddev = (
            self.softplus(raw_noise_stddev) + self.noise_epsilon
        ) * self.training
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        logits = noisy_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.num_expert), dim=1
        )
        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.top_k < self.num_expert:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)

        return (
            top_k_indices.contiguous().view(-1),
            top_k_gates.contiguous().unsqueeze(1),
            loss,
        )
