r"""
Patching some of Megatron-LM's functions to create an MoE model
"""
import torch

def patch_loss_func_v2_5(loss_func):
    r"""
    Patch model's loss_func to support balance loss
    """

    from megatron.mpu import is_pipeline_last_stage
    from megatron.mpu import get_tensor_model_parallel_group
    from megatron import get_args
    from megatron import get_num_microbatches

    if not get_args().balance_strategy:
        return loss_func

    def loss_func_with_balance_loss(model, output_tensor):
        args = get_args()
        assert args.balance_strategy, "Only use patched loss_func when having balance_strategy."
        assert is_pipeline_last_stage(), "Only call loss_func at pipeline last stage."
        
        output = loss_func(output_tensor)
        
        while hasattr(model, 'module'):
            model = model.module

        loss_list = [l.mlp.gate.get_loss(clear=False).view(1)
                for l in model.language_model.encoder.layers
                if l.mlp.gate.has_loss]

        if hasattr(model.language_model, "decoder") and model.language_model.decoder is not None:
            loss_list_decoder = [l.mlp.gate.get_loss(clear=False).view(1)
                    for l in model.language_model.decoder.layers
                    if l.mlp.gate.has_loss]
            loss_list.append(loss_list_decoder)
            
        if len(loss_list) == 0:
            return output

        loss_name = args.balance_strategy + "_loss"
        (loss, state_dict), bal_loss = (
            output,
            torch.cat(loss_list).mean() * args.balance_loss_weight / args.pipeline_model_parallel_size
        )

        bal_loss = bal_loss / get_num_microbatches()

        # avarage across moe group
        moe_group = get_tensor_model_parallel_group()
        world_size = torch.distributed.get_world_size(group=moe_group)
        averaged_bal_loss = bal_loss.clone().detach()
        torch.distributed.all_reduce(averaged_bal_loss, group=moe_group)
        averaged_bal_loss /= world_size

        loss += bal_loss
        state_dict[loss_name] = averaged_bal_loss

        return loss, state_dict

    return loss_func_with_balance_loss

def patch_forward_step(forward_step_func, Megatron_Version="v2.2"):
    r"""
    Patch model's forward_step_func to support balance loss
    """

    from megatron.mpu import is_pipeline_last_stage
    from megatron.mpu import get_tensor_model_parallel_group
    from megatron import get_args

    if not get_args().balance_strategy:
        return forward_step_func

    def forward_step_with_balance_loss_v2_2(data_iterator, model, input_tensor):
        args = get_args()
        output = forward_step_func(data_iterator, model, input_tensor)

        if not is_pipeline_last_stage() or not args.balance_strategy:
            return output

        while hasattr(model, 'module'):
            model = model.module

        loss_list = [l.mlp.gate.get_loss(clear=False).view(1)
                for l in model.language_model.transformer.layers
                if l.mlp.gate.has_loss]
        if len(loss_list) == 0:
            return output

        loss_name = args.balance_strategy + "_loss"
        (loss, state_dict), bal_loss = (
            output,
            torch.cat(loss_list).mean() * args.balance_loss_weight
        )

        # avarage across moe group
        moe_group = get_tensor_model_parallel_group()
        world_size = torch.distributed.get_world_size(group=moe_group)
        averaged_bal_loss = bal_loss.clone().detach()
        torch.distributed.all_reduce(averaged_bal_loss, group=moe_group)
        averaged_bal_loss /= world_size

        loss += bal_loss
        state_dict[loss_name] = averaged_bal_loss

        return loss, state_dict

    def forward_step_with_balance_loss_v2_5(data_iterator, model):
        from functools import partial
        output, loss_func = forward_step_func(data_iterator, model)
    
        while hasattr(model, 'module'):
            model = model.module

        loss_list = [l.mlp.gate.get_loss(clear=False).view(1)
                for l in model.language_model.encoder.layers
                if l.mlp.gate.has_loss]

        bal_loss = torch.cat(loss_list).mean() * get_args().balance_loss_weight / get_args().pipeline_model_parallel_size
        return output, partial(patch_loss_func_v2_5(loss_func), model), bal_loss

    if Megatron_Version == "v2.2":
        return forward_step_with_balance_loss_v2_2
    elif Megatron_Version == "v2.5":
        return forward_step_with_balance_loss_v2_5
    elif Megatron_Version == "v3.0.2":
        return forward_step_with_balance_loss_v2_5
    else:
        assert False, f"megatron version {Megatron_Version} not known."



def patch_model_provider(model_provider, gate=None, Megatron_Version='v2.2'):
    from megatron import get_args

    def fmoefied_model_provider_v2_2():
        from .layers import fmoefy
        args = get_args()
        hhs = args.hidden_size * 4
        assert hhs % args.top_k == 0
        hhs = hhs // args.top_k
        assert hhs % args.tensor_model_parallel_size == 0
        hhs = hhs // args.tensor_model_parallel_size
        return fmoefy(
            model_provider(),
            fmoe_num_experts=args.fmoe_num_experts,
            hidden_hidden_size=hhs,
            top_k=args.top_k,
            gate=gate,
            megatron_version="v2.2"
        )
    
    def fmoefied_model_provider_v2_5(pre_process, post_process):
        from .layers import fmoefy
        args = get_args()
        hhs = args.hidden_size * 4
        assert hhs % args.top_k == 0
        hhs = hhs // args.top_k
        assert hhs % args.tensor_model_parallel_size == 0
        hhs = hhs // args.tensor_model_parallel_size
        return fmoefy(
            model_provider(pre_process=pre_process, post_process=post_process),
            fmoe_num_experts=args.fmoe_num_experts,
            hidden_hidden_size=hhs,
            top_k=args.top_k,
            gate=gate,
            megatron_version="v2.5"
        )
    
    def fmoefied_model_provider_v3_0_2(pre_process, post_process):
        from .layers import fmoefy
        args = get_args()
        hhs = args.hidden_size * 4
        assert hhs % args.top_k == 0
        hhs = hhs // args.top_k
        assert hhs % args.tensor_model_parallel_size == 0
        hhs = hhs // args.tensor_model_parallel_size
        return fmoefy(
            model_provider(pre_process=pre_process, post_process=post_process),
            fmoe_num_experts=args.fmoe_num_experts,
            hidden_hidden_size=hhs,
            top_k=args.top_k,
            gate=gate,
            megatron_version="v3.0.2"
        )

    if Megatron_Version == 'v2.2':
        return fmoefied_model_provider_v2_2
    elif Megatron_Version == 'v2.5':
        return fmoefied_model_provider_v2_5
    elif Megatron_Version == 'v3.0.2':
        return fmoefied_model_provider_v3_0_2
    else:
        assert False, f"Megatron Version {Megatron_Version} unknown."
