r"""
The fmoe.functions module contains functions that are directly warped up from
C/CUDA functions to complete distributed communication, computation and gradient
computation.
"""

import torch
from torch.autograd import Function
import fmoe_cuda
from .utils import get_torch_default_comm


def moe_prepare_forward(gate, num_expert, world_size, comm=None):
    r"""
    Prepare necessary information from gate output for MoE computation.

    Args:
        gate: a 1-d Long Tensor representing the target expert of each input
        sample.
        num_expert: number of experts on each worker.
        world_size: number of workers that hold different experts.
        comm: the communicator of all workers in the expert-parallel group.
    """
    if world_size > 1:
        if comm is None:
            comm = get_torch_default_comm()
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
        fwd_expert_count = global_expert_count.view(world_size,
                num_expert).sum(dim=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())
    return (
        pos,
        local_expert_count.cpu(),
        global_expert_count.cpu(),
        fwd_expert_count.cpu(),
        fwd_batch_size,
    )


class MOEScatter(Function):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """

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
        ctx.moe_args = inp.shape[0], world_size
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, global_grad_in):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensors
        (local_batch_size, world_size) = ctx.moe_args

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
    r"""
    Computes linear operators within one GPU on different experts simutaneously.
    """

    @staticmethod
    def forward(ctx, global_input_buf, fwd_expert_count, weight, bias=None):
        (global_output_buf,) = fmoe_cuda.forward(
            global_input_buf, fwd_expert_count, weight, bias
        )
        variables = (global_input_buf, fwd_expert_count, weight, bias)
        ctx.save_for_backward(*variables)
        return global_output_buf

    @staticmethod
    def backward(ctx, grad_out):
        (input_buf, fwd_expert_count, weight, bias) = ctx.saved_tensors
        grad_inp_buf, grad_weight, grad_bias = fmoe_cuda.backward(
            grad_out, input_buf, fwd_expert_count, weight, bias
        )

        if not torch.is_tensor(bias):
            grad_bias = None

        return grad_inp_buf, None, grad_weight, grad_bias


class MOEGather(Function):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MOEScatter.
    """

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

        ctx.moe_args = (global_output_buf.shape[0], world_size)
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensors
        fwd_batch_size, world_size = ctx.moe_args
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


class AllGather(Function):
    r"""
    A wrapper for the All-Gather function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        tensor_list = [torch.empty_like(inp) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, inp, group=group)
        torch.cuda.synchronize()
        output = torch.cat(tensor_list, dim=0)
        ctx.args = rank, inp.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rank, dim0 = ctx.args
        return grad_out[rank * dim0 : (rank + 1) * dim0], None, None, None


class Slice(Function):
    r"""
    A wrapper for the Slice function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        B: int = inp.shape[0]
        local_batch_size = B // world_size
        batch_start = local_batch_size * rank
        batch_end = min(batch_start + local_batch_size, B)
        inp = inp[batch_start:batch_end]
        ctx.args = world_size, group
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        world_size, group = ctx.args
        tensor_list = [torch.empty_like(grad_out) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, grad_out, group=group)
        torch.cuda.synchronize()
        grad_out = torch.cat(tensor_list, dim=0)
        return grad_out, None, None, None
