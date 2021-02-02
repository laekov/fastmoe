import torch
from torch.autograd import Function
import fmoe_cuda


def moe_prepare_forward(gate, num_expert, world_size, comm=None):
    if comm is None:
        comm = torch.distributed.distributed_c10d._default_pg
    if world_size > 1:
        fmoe_cuda.ensure_nccl(comm, gate)

    with torch.no_grad():
        _, pos = torch.sort(gate)
        gate_idx, gate_count = torch.unique(gate, return_counts=True)
        local_expert_count = torch.zeros(
            num_expert * world_size, device=gate.device, dtype=torch.long
        )
        local_expert_count.index_put_((gate_idx.long(),), gate_count)

        if world_size > 1:
            (global_expert_count,) = fmoe_cuda.expert_exchange(
                local_expert_count, num_expert, world_size
            )
        else:
            global_expert_count = local_expert_count
        fwd_expert_count = global_expert_count.view(world_size, num_expert).sum(dim=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())
    return (
        pos,
        local_expert_count.cpu(),
        global_expert_count.cpu(),
        fwd_expert_count.cpu(),
        fwd_batch_size,
    )


class MOEScatter(Function):
    @staticmethod
    def forward(
        ctx,
        inp,
        pos,
        local_expert_count,
        global_expert_count,
        fwd_batch_size,
        world_size,
    ):
        (local_input_buf,) = fmoe_cuda.local_scatter(inp, pos)
        if world_size > 1:
            (global_input_buf,) = fmoe_cuda.global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
            )
        else:
            global_input_buf = local_input_buf
        ctx.moe_args = fwd_batch_size, inp.shape[0], world_size
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, global_grad_in):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensors
        (fwd_batch_size, local_batch_size, world_size) = ctx.moe_args

        if world_size > 1:
            (local_grad_in,) = fmoe_cuda.global_gather(
                global_grad_in,
                local_expert_count,
                global_expert_count,
                local_batch_size,
                world_size,
            )
        else:
            local_grad_in = global_grad_in
        (grad_in,) = fmoe_cuda.local_gather(local_grad_in, pos)
        return grad_in, None, None, None, None, None


class MOELinear(Function):
    @staticmethod
    def forward(ctx, global_input_buf, weight, fwd_expert_count):
        (global_output_buf,) = fmoe_cuda.forward(
            global_input_buf, weight, fwd_expert_count
        )
        variables = (global_input_buf, weight, fwd_expert_count)
        ctx.save_for_backward(*variables)
        return global_output_buf

    @staticmethod
    def backward(ctx, grad_out):
        (input_buf, weight, fwd_expert_count) = ctx.saved_tensors
        grad_inp_buf, grad_weight = fmoe_cuda.backward(
            grad_out, input_buf, weight, fwd_expert_count
        )
        return grad_inp_buf, grad_weight, None


class MOEGather(Function):
    @staticmethod
    def forward(
        ctx,
        global_output_buf,
        pos,
        local_expert_count,
        global_expert_count,
        local_batch_size,
        world_size,
    ):
        if world_size > 1:
            (local_output_buf,) = fmoe_cuda.global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                local_batch_size,
                world_size,
            )
        else:
            local_output_buf = global_output_buf
        (output,) = fmoe_cuda.local_gather(local_output_buf, pos)

        ctx.moe_args = local_batch_size, global_output_buf.shape[0], world_size
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensors
        local_batch_size, fwd_batch_size, world_size = ctx.moe_args
        (grad_out_buf,) = fmoe_cuda.local_scatter(grad_out.contiguous(), pos)
        if world_size > 1:
            (global_grad_out_buf,) = fmoe_cuda.global_scatter(
                grad_out_buf,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
            )
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None, None, None
