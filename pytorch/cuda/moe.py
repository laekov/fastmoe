import math
from torch import nn
from torch.autograd import Function
import torch

import moe_cuda

torch.manual_seed(42)


class MOEFunction(Function):
    @staticmethod
    def forward(ctx, input, gate, weight):
        output = moe_cuda.forward(input, gate, weight)
        variables = [input, gate, weight]
        ctx.save_for_backward(*variables)

        return output[0]

    @staticmethod
    def backward(ctx, grad_out):
        grad_input, grad_weight = moe_cuda.backward(
            grad_out.contiguous(), *ctx.saved_variables)
        return grad_input, grad_weight


class MOELayer(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=4096):
        super(MOELayer, self).__init__()
        self.weight = nn.Parameter(
            torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input, gate):
        return MOEFunction.apply(input, gate, self.weight)


batch_size = 64
num_expert = 32
in_feat = 512
out_feat = 512

moe = MOELayer(num_expert, in_feat, out_feat).cuda()

input = torch.rand(batch_size, in_feat).cuda()
gate = torch.randint(low=0, high=num_expert, size=(batch_size, )).int().cuda()

output = moe(input, gate)
