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
    # Fill in this function to conduct balance related jobs
    pass
