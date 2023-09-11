r"""
The smart schedule proposed in FasterMoE.
"""
import torch
from torch.autograd.function import Function

from fmoe.functions import prepare_forward, ensure_comm
from fmoe.functions import _local_scatter, _local_gather 
import fmoe_cuda as fmoe_native
from fmoe.fastermoe import expert_utils

from .shadow_policy import get_shadow_policy


class MoEForward(Function):
    @staticmethod
    def forward(
            ctx,
            expert_fn,
            experts,
            inp, # models,
            pos_s, pos_g,
            local_expert_count, global_expert_count,
            stored_models,
            fwd_batch_size, out_batch_size,
            num_expert,
            world_size):
        local_input_buf = _local_scatter(inp, pos_s)

        ctx.gibs = [None] * (world_size * num_expert * 2)
        ctx.gobs = [None] * (world_size * num_expert * 2)
        def _expert_forward(x, y, expert_idx, store_idx):
            nothing = lambda a: a
            x = x.data
            with torch.enable_grad():
                x.requires_grad = True
                try:
                    # To skip torch autograd's version check.
                    with torch.autograd.graph.saved_tensors_hooks(nothing, nothing):
                        y0 = expert_fn(x, torch.tensor([x.shape[0]], dtype=torch.int64), expert_idx)
                except Exception as e:
                    # Ignore the error and fall back for compatibility to older
                    # versions of PyTorch
                    y0 = expert_fn(x, torch.tensor([x.shape[0]], dtype=torch.int64), expert_idx)
            ctx.gibs[store_idx] = x
            ctx.gobs[store_idx] = y0
            y.copy_(y0)

        ctx.experts = experts
        if stored_models.any():
            ctx.expert_size = expert_utils.get_expert_param_size(experts, 0)
            for i in range(num_expert):
                assert ctx.expert_size == expert_utils.get_expert_param_size(experts, i), "report bug"            
        else:
            ctx.expert_size = 0
        get_param_fn = lambda out, idx: expert_utils.get_expert_params(experts, out, idx)
        pop_fn = lambda idx: expert_utils.pop_expert_params(experts, idx)
        ctx.shadows = [None] * world_size * num_expert
        def stash_fn(params, store_idx, expert_idx):
            expert_utils.stash_expert_params(experts, params, expert_idx)
            ctx.shadows[store_idx] = params

        local_output_buf, gib = fmoe_native.smart_sch_forward(
                local_input_buf,
                local_expert_count, global_expert_count, 
                stored_models, fwd_batch_size, ctx.expert_size,
                world_size, _expert_forward, get_param_fn, stash_fn, pop_fn)

        out = _local_gather(local_output_buf, pos_g, out_batch_size,
                maybe_overlap=False)
        
        # gib and local_input_buf are necessary, because ctx.gibs are created
        # based on their memory
        variables = (pos_s, pos_g, local_expert_count, global_expert_count,
                stored_models, gib, local_input_buf)
        
        ctx.moe_args = fwd_batch_size, inp.shape[0], num_expert, world_size
        ctx.save_for_backward(*variables)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        (pos_s, pos_g, local_expert_count, global_expert_count,
                stored_models, _1, _2) = ctx.saved_tensors
        (fwd_batch_size, inp_batch_size, num_expert, world_size) = ctx.moe_args

        def _expert_backward(grad_y, grad_x, expert_idx, store_idx):
            y = ctx.gobs[store_idx]
            x = ctx.gibs[store_idx]
            torch.autograd.backward([y], [grad_y])
            grad_x.copy_(x.grad)

        experts = ctx.experts
        def stash_fn(store_idx, expert_idx):
            expert_utils.stash_expert_params(experts, ctx.shadows[store_idx], expert_idx)
        pop_fn = lambda idx: expert_utils.pop_expert_params(experts, idx)
        def collect_fn(store_idx, root, expert_idx): 
            grad = ctx.shadows[store_idx]
            expert_utils.collect_expert_grads(experts, grad, expert_idx)
            fmoe_native.reduce_grad(grad, root, ctx.expert_size)
        set_grad_fn = lambda store_idx, expert_idx: expert_utils.set_grads(experts, ctx.shadows[store_idx], expert_idx)

        grad_out_buf = _local_scatter(grad_out.contiguous(), pos_g)
        grad_in_buf = fmoe_native.smart_sch_backward(
                grad_out_buf,
                local_expert_count, global_expert_count,
                stored_models,
                pos_s.shape[0], fwd_batch_size,
                world_size,
                _expert_backward, stash_fn, pop_fn, collect_fn, set_grad_fn)
        grad_in = _local_gather(grad_in_buf, pos_s, inp_batch_size)

        return (None, None, grad_in, None, None, None, None, None, None, None, None, None)


policy_fn = None


def _fmoe_general_global_forward(inp, gate, expert_fn, n_expert, world_size, experts=None, stored_models=None):
    # TODO: Using multiple tensors as input is to be supported.
    assert(isinstance(inp, torch.Tensor))
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, n_expert, world_size)

    global policy_fn
    if policy_fn is None:
        policy_fn = get_shadow_policy(d_model=inp.shape[-1])

    if stored_models is None:
        stored_models = policy_fn(local_expert_count, global_expert_count,
                n_expert, world_size)

    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]
    out_batch_size = inp.shape[0] * topk

    return MoEForward.apply(expert_fn, experts, inp,
            torch.div(pos, topk, rounding_mode='floor'), pos,
            local_expert_count, global_expert_count, stored_models,
            fwd_batch_size, out_batch_size, n_expert, world_size)
