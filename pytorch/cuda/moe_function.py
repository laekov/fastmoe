import torch
from torch.autograd import Function
import moe_cuda


class MOELocal(Function):
    @staticmethod
    def forward(ctx, inp, gate, weight):
        expert_count, pos = moe_cuda.expert_count(gate, weight.shape[0])
        input_buf, = moe_cuda.local_scatter(inp, pos)
        output_buf, = moe_cuda.forward(input_buf, weight, expert_count)
        output = moe_cuda.local_gather(output_buf, pos)

        variables = [input_buf, gate, weight, expert_count, pos]
        ctx.save_for_backward(*variables)

        return output[0]

    @staticmethod
    def backward(ctx, grad_out):
        input_buf, gate, weight, expert_count, pos = ctx.saved_tensors

        grad_out_buf, = moe_cuda.local_scatter(grad_out.contiguous(), pos)
        grad_inp_buf, grad_weight = moe_cuda.backward(
                grad_out_buf, input_buf, weight, expert_count)
        grad_inp, = moe_cuda.local_gather(grad_inp_buf, pos)

        return grad_inp, None, grad_weight


class MOEGlobal(Function):
    @staticmethod
    def forward(ctx, inp, gate, weight, world_size):
        num_expert = weight.shape[0]

        local_expert_count, pos = moe_cuda.expert_count(gate, 
                world_size * num_expert)
        global_expert_count, fwd_expert_count = moe_cuda.expert_exchange(
                local_expert_count, num_expert, world_size)
        fwd_batch_size = int(fwd_expert_count.sum().item())

        local_input_buf, = moe_cuda.local_scatter(inp, pos)
        global_input_buf, = moe_cuda.global_scatter(local_input_buf, 
                local_expert_count, global_expert_count,
                fwd_batch_size, world_size)

        global_output_buf, = moe_cuda.forward(global_input_buf, weight, 
                fwd_expert_count)

        local_output_buf, = moe_cuda.global_gather(global_output_buf,
                local_expert_count, global_expert_count,
                inp.shape[0], world_size)
        output, = moe_cuda.local_gather(local_output_buf, pos)

        variables = (global_input_buf, gate, weight, 
                local_expert_count, global_expert_count, fwd_expert_count,
                pos)
        ctx.moe_args = (num_expert, inp.shape[0], fwd_batch_size, world_size)
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        (input_buf, gate, weight, 
                local_expert_count, global_expert_count, fwd_expert_count, 
                pos) = ctx.saved_tensors
        num_expert, local_batch_size, fwd_batch_size, world_size = ctx.moe_args

        grad_out_buf, = moe_cuda.local_scatter(grad_out.contiguous(), pos)
        global_grad_out_buf, = moe_cuda.global_scatter(grad_out_buf,
                local_expert_count, global_expert_count,
                fwd_batch_size, world_size)

        grad_inp_buf, grad_weight = moe_cuda.backward(
                global_grad_out_buf, input_buf, weight, fwd_expert_count)

        local_grad_inp_buf, = moe_cuda.global_gather(grad_inp_buf,
                local_expert_count, global_expert_count,
                local_batch_size, world_size)
        grad_inp, = moe_cuda.local_gather(local_grad_inp_buf, pos)

        return grad_inp, None, grad_weight, None


def moe(inp, gate, weight, world_size):
    if world_size is not None:
        return MOEGlobal.apply(inp, gate, weight, world_size)
    else:
        return MOELocal.apply(inp, gate, weight)
