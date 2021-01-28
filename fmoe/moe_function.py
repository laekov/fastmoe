import torch
from torch.autograd import Function
import fmoe_cuda


class MOELocal(Function):
    @staticmethod
    def forward(ctx, inp, gate, weight):
        _, pos = torch.sort(gate)
        gate_idx, gate_count = torch.unique(gate, return_counts=True)
        expert_count = torch.zeros(weight.shape[0], device=weight.device, 
                dtype=torch.long)
        expert_count.index_put_((gate_idx.long(), ), gate_count)

        # expert_count, pos = fmoe_cuda.expert_count(gate, weight.shape[0])
        ecc = expert_count.cpu()
        input_buf, = fmoe_cuda.local_scatter(inp, pos)
        output_buf, = fmoe_cuda.forward(input_buf, weight, ecc)
        output = fmoe_cuda.local_gather(output_buf, pos)

        variables = [input_buf, gate, weight, ecc, pos]
        ctx.save_for_backward(*variables)

        return output[0]

    @staticmethod
    def backward(ctx, grad_out):
        input_buf, gate, weight, expert_count, pos = ctx.saved_tensors

        grad_out_buf, = fmoe_cuda.local_scatter(grad_out.contiguous(), pos)
        grad_inp_buf, grad_weight = fmoe_cuda.backward(
                grad_out_buf, input_buf, weight, expert_count)
        grad_inp, = fmoe_cuda.local_gather(grad_inp_buf, pos)

        return grad_inp, None, grad_weight


class MOEGlobal(Function):
    @staticmethod
    def forward(ctx, inp, gate, weight, world_size):
        num_expert = weight.shape[0]

        local_expert_count, pos = fmoe_cuda.expert_count(gate, 
                world_size * num_expert)
        global_expert_count, fwd_expert_count = fmoe_cuda.expert_exchange(
                local_expert_count, num_expert, world_size)
        fwd_batch_size = int(fwd_expert_count.sum().item())

        local_input_buf, = fmoe_cuda.local_scatter(inp, pos)

        local_output_buf, global_input_buf = fmoe_cuda.global_fused_forward(
                local_input_buf, weight,
                local_expert_count, global_expert_count,
                fwd_batch_size, inp.shape[0], world_size)

        output, = fmoe_cuda.local_gather(local_output_buf, pos)

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

        grad_out_buf, = fmoe_cuda.local_scatter(grad_out.contiguous(), pos)
        global_grad_out_buf, = fmoe_cuda.global_scatter(grad_out_buf,
                local_expert_count, global_expert_count,
                fwd_batch_size, world_size)

        grad_inp_buf, grad_weight = fmoe_cuda.backward(
                global_grad_out_buf, input_buf, weight, fwd_expert_count)

        local_grad_inp_buf, = fmoe_cuda.global_gather(grad_inp_buf,
                local_expert_count, global_expert_count,
                local_batch_size, world_size)
        grad_inp, = fmoe_cuda.local_gather(local_grad_inp_buf, pos)

        return grad_inp, None, grad_weight, None


def moe(inp, gate, weight, world_size):
    if world_size is not None and world_size > 1:
        return MOEGlobal.apply(inp, gate, weight, world_size)
    else:
        return MOELocal.apply(inp, gate, weight)
