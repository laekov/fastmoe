r"""
The smart schedule proposed in FasterMoE.
"""
import torch
from torch.autograd.function import Function

from fmoe.functions import prepare_forward, ensure_comm
from fmoe.functions import _local_scatter, _local_gather 
import fmoe_cuda as fmoe_native
import expert_utils


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
            world_size):
        local_input_buf = _local_scatter(inp, pos_s)

        # TODO: leave this for furture work of expert shadowing
        # model_params = [[tuple(m.parameters()) for m in node] for node in models]

        ctx.gibs = [None] * (world_size * 2)
        ctx.gobs = [None] * (world_size * 2)
        def _expert_forward(x, y, idx):
            x = x.data
            with torch.enable_grad():
                x.requires_grad = True
                y0 = expert_fn(x, [x.shape[0]])
            ctx.gibs[idx] = x
            ctx.gobs[idx] = y0
            y.copy_(y0)

        ctx.experts = experts
        if stored_models.any():
            ctx.expert_size = expert_utils.get_expert_param_size(experts)
        else:
            ctx.expert_size = 0
        get_param_fn = lambda out: expert_utils.get_expert_params(experts, out)
        pop_fn = lambda: expert_utils.pop_expert_params(experts)
        ctx.shadows = [None] * world_size
        def stash_fn(params, idx):
            expert_utils.stash_expert_params(experts, p)
            ctx.shadows[idx] = params

        local_output_buf, gib = fmoe_native.smart_sch_forward(
                local_input_buf,
                local_expert_count, global_expert_count, 
                stored_models, fwd_batch_size, ctx.expert_size,
                world_size, _expert_forward, get_param_fn, stash_fn, pop_fn)

        out = _local_gather(local_output_buf, pos_g, out_batch_size,
                maybe_overlap=False)
        
        variables = (pos_s, pos_g, local_expert_count, global_expert_count,
                stored_models, gib)
        
        ctx.moe_args = fwd_batch_size, inp.shape[0], world_size
        ctx.save_for_backward(*variables)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        (pos_s, pos_g, local_expert_count, global_expert_count,
                stored_models, _) = ctx.saved_tensors
        (fwd_batch_size, inp_batch_size, world_size) = ctx.moe_args

        def _expert_backward(grad_y, grad_x, idx):
            y = ctx.gobs[idx]
            torch.autograd.backward([y], [grad_y])
            x = ctx.gibs[idx]
            grad_x.copy_(x.grad)

        experts = ctx.experts
        def stash_fn(idx):
            expert_utils.stash_expert_params(experts, ctx.shadows[idx])
        pop_fn = lambda: expert_utils.pop_expert_params(experts)
        collect_fn = lambda g: expert_utils.collect_expert_grads(experts, g)
        set_grad_fn = lambda g: expert_utils.set_grads(experts, g)

        grad_out_buf = _local_scatter(grad_out.contiguous(), pos_g)
        grad_in_buf = fmoe_native.smart_sch_backward(
                grad_out_buf,
                local_expert_count, global_expert_count,
                stored_models,
                pos_s.shape[0], fwd_batch_size, ctx.expert_size,
                world_size, _expert_backward,
                stash_fn, pop_fn, collect_fn, set_grad_fn)
        grad_in = _local_gather(grad_in_buf, pos_s, inp_batch_size)

        return (None, None, grad_in, None, None, None, None, None, None, None, None)


def _fmoe_general_global_forward(inp, gate, expert_fn, n_expert, world_size, experts=None):
    # TODO: Using multiple tensors as input is to be supported.
    assert(isinstance(inp, torch.Tensor))
    # TODO: Support many experts on each process
    assert(n_expert == 1)
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, n_expert, world_size)

    # TODO: Expert shadowing is to be supported. Currently using all 0s
    stored_models = torch.zeros(n_expert * world_size, dtype=torch.bool)

    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]
    out_batch_size = inp.shape[0] * topk

    return MoEForward.apply(expert_fn, experts, inp,
            torch.div(pos, topk, rounding_mode='floor'), pos,
            local_expert_count, global_expert_count, stored_models,
            fwd_batch_size, out_batch_size, world_size)
