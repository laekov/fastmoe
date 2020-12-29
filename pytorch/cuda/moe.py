import math
from torch import nn
from torch.autograd import Function
import torch

import moe_cuda

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class MOEFunction(Function):
    @staticmethod
    def forward(ctx, inp, gate, weight):
        output = moe_cuda.forward(inp, gate, weight)
        variables = [inp, gate, weight]
        ctx.save_for_backward(*variables)

        return output[0]

    @staticmethod
    def backward(ctx, grad_out):
        grad_inp, grad_weight = moe_cuda.backward(
            grad_out.contiguous(), *ctx.saved_tensors)
        return grad_inp, None, grad_weight


class MOELayer(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=4096):
        super(MOELayer, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(
            torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, out_features=out_feat)
            self.weight.data[i] = linear.weight.data

    def forward(self, inp, gate):
        return MOEFunction.apply(inp, gate, self.weight)


class MOELayer_einsum(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=4096):
        super(MOELayer_einsum, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(
            torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, out_features=out_feat)
            self.weight.data[i] = linear.weight.data
    
    def forward(self, inp, gate):
        gate_long = gate.long()
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.out_feat))
        for i in range(batch_size):
            x[i] = self.weight[gate_long[i]] @ inp[i]
        return x

batch_size = 1
num_expert = 1
in_feat = 3
out_feat = 3

moe = MOELayer(num_expert, in_feat, out_feat).cuda()
moe_einsum = MOELayer_einsum(num_expert, in_feat, out_feat).cuda()
moe_einsum.weight.data = moe.weight.data.clone()


inp = torch.rand(batch_size, in_feat).cuda()
gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda()

print(inp.type())
print(moe.weight.data.type())

print(inp)
print(gate)
output = moe(inp, gate)

print(inp)
print(gate)
output_einsum = moe_einsum(inp.clone(), gate.clone())

print(output)
print(output_einsum)

#y = output.mean()
#y.backward()