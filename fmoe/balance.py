import torch
import torch.nn.functional as F

metrics = {
    "coefficient-variation": lambda c_e: torch.std(c_e) / torch.mean(c_e),
    "Lmax-over-Lmin": lambda c_e: (torch.max(c_e) + 1) / (torch.min(c_e) + 1),
    "Lmax-over-Lmean": lambda c_e: torch.max(c_e) / torch.mean(c_e),
}


def reset_balance_profile(balance_dict, num_layers, balance_strategy):
    for key in metrics:
        balance_dict[key] = [None for _ in range(num_layers)]
    if balance_strategy:
        balance_dict[f"{balance_strategy}_loss"] = [None for _ in range(num_layers)]


def update_balance_profile(
    balance_dict,
    gate_top_k_idx,
    _gate_score_top_k,
    gate_context,
    layer_idx,
    num_expert,
    balance_strategy,
):
    c_e = torch.scatter_add(
        torch.zeros(num_expert, device=gate_top_k_idx.device),
        0,
        gate_top_k_idx,
        torch.ones_like(gate_top_k_idx, dtype=torch.float),
    )
    for key in metrics:
        balance_dict[key][layer_idx] = metrics[key](c_e)
    S = gate_top_k_idx.shape[0]
    if balance_strategy == "gshard":
        gate_score_all = gate_context
        m_e = torch.sum(F.softmax(gate_score_all, dim=1), dim=0) / S
        balance_dict["gshard_loss"][layer_idx] = torch.sum(c_e * m_e) / num_expert / S
    elif balance_strategy == "noisy":
        balance_dict["noisy_loss"][layer_idx] = gate_context
