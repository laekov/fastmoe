r"""
The fmoe.functions module contains functions that are directly warped up from
C/CUDA functions to complete distributed communication, computation and gradient
computation.
"""

import torch
from torch.autograd import Function
import fmoe_cuda
from .utils import get_torch_default_comm


def ensure_comm(t, comm):
    if comm is None:
        comm = get_torch_default_comm()
    fmoe_cuda.ensure_nccl(comm, t)


def count_by_gate(gate, num_expert, world_size, require_pos=True):
    with torch.no_grad():
        local_expert_count = torch.zeros(
            num_expert * world_size, device=gate.device, dtype=torch.int32
        )
        fmoe_cuda.expert_count(gate, local_expert_count)
        local_expert_count = local_expert_count.long()

        if world_size > 1:
            global_expert_count = fmoe_cuda.expert_exchange(
                local_expert_count, num_expert, world_size
            )
        else:
            global_expert_count = local_expert_count
        if not require_pos:
            pos = None
        else:
            lec_cum = torch.cumsum(local_expert_count, dim=0).int()
            pos_size = lec_cum[-1].item()
            pos = torch.empty((pos_size,), device=gate.device, dtype=torch.long)
            fmoe_cuda.assign_pos(lec_cum, gate, pos)
    return pos, local_expert_count, global_expert_count


def prepare_forward(gate, num_expert, world_size):
    r"""
    Prepare necessary information from gate output for MoE computation.

    Args:
        gate: a 1-d Long Tensor representing the target expert of each input
        sample.
        num_expert: number of experts on each worker.
        world_size: number of workers that hold different experts.
        comm: the communicator of all workers in the expert-parallel group.
    """
    pos, local_expert_count, global_expert_count = count_by_gate(gate, 
            num_expert, world_size)
    with torch.no_grad():
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


def _local_scatter(inp, pos):
    inp_buf = torch.index_select(inp, 0, pos)
    return inp_buf


def _local_gather(inp, pos, out_batch_size, maybe_overlap=True):
    inp_buf = torch.zeros(out_batch_size, inp.shape[-1],
            dtype=inp.dtype, device=inp.device)
    if maybe_overlap:
        inp_buf.index_add_(0, pos, inp)
    else:
        inp_buf.index_copy_(0, pos, inp)
    return inp_buf


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
        local_input_buf = _local_scatter(inp, pos)
        if world_size > 1:
            global_input_buf = fmoe_cuda.global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
            )
        else:
            global_input_buf = local_input_buf
        ctx.moe_args = inp.shape[0], pos.shape[0], world_size
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, global_grad_in):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensors
        (inp_batch_size, buf_batch_size, world_size) = ctx.moe_args

        if world_size > 1:
            local_grad_in = fmoe_cuda.global_gather(
                global_grad_in,
                local_expert_count,
                global_expert_count,
                buf_batch_size,
                world_size,
            )
        else:
            local_grad_in = global_grad_in
        grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        return grad_in, None, None, None, None, None

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
            local_output_buf = fmoe_cuda.global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                pos.shape[0],
                world_size,
            )
        else:
            local_output_buf = global_output_buf
        output = _local_gather(local_output_buf, pos, local_batch_size,
                maybe_overlap=False)

        ctx.moe_args = (global_output_buf.shape[0], world_size)
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensors
        fwd_batch_size, world_size = ctx.moe_args
        grad_out_buf = _local_scatter(grad_out.contiguous(), pos)
        if world_size > 1:
            global_grad_out_buf = fmoe_cuda.global_scatter(
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
