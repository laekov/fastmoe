r"""
Support for monitoring loss in Megatron
"""
import torch
from fmoe.balance import reset_balance_profile
from fmoe.balance import update_balance_profile
from fmoe.utils import get_torch_default_comm


balance_dict = {}
num_layers = 0


def reset_gate_hook(_num_layers=None):
    from megatron import get_args

    global balance_dict, num_layers
    if _num_layers is not None:
        num_layers = _num_layers
    reset_balance_profile(balance_dict, num_layers, get_args().balance_strategy)


def get_balance_profile():
    global balance_dict
    return balance_dict


def generate_megatron_gate_hook(layer_idx, num_expert_global):
    from megatron import get_args

    balance_strategy = get_args().balance_strategy

    def megatron_gate_hook(gate_top_k_idx, gate_score_top_k, gate_context):
        global balance_dict
        update_balance_profile(
            balance_dict,
            gate_top_k_idx,
            gate_score_top_k,
            gate_context,
            layer_idx,
            num_expert_global,
            balance_strategy,
        )

    return megatron_gate_hook


def add_balance_log(model, writer, iteration):
    from megatron import is_last_rank

    while hasattr(model, 'module'):
        model = model.module

    losses = [l.mlp.gate.get_loss(clear=True)
            for l in model.language_model.transformer.layers
            if l.mlp.gate.has_loss]
    if len(losses) == 0:
        return
    balance_dict_tensor = torch.vstack(losses).detach()
    world_group = get_torch_default_comm()
    world_size = torch.distributed.get_world_size(group=world_group)
    torch.distributed.all_reduce(balance_dict_tensor, group=world_group)
    balance_dict_tensor /= world_size

    if writer and is_last_rank():
        for idx, metric_name in enumerate(balance_dict):
            for layer_id, val in enumerate(balance_dict_tensor[idx]):
                writer.add_scalar(
                    f"balance-{metric_name}/layer-{layer_id}", val.item(), iteration
                )
            writer.add_scalar(
                f"balance-{metric_name}/all",
                balance_dict_tensor[idx].mean().item(),
                iteration,
            )
