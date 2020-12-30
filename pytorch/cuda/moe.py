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
        out_feat, in_feat = weight.size()[1:]
        weight_column_major = weight.transpose(-1, -2).contiguous().view(-1, out_feat, in_feat)
        output = moe_cuda.forward(inp, gate, weight_column_major)
        variables = [inp, gate, weight_column_major]
        ctx.save_for_backward(*variables)

        return output[0]

    @staticmethod
    def backward(ctx, grad_out):
        print("grad_out", grad_out)
        print("input", ctx.saved_tensors[0])
        grad_inp, grad_weight = moe_cuda.backward(
            grad_out.contiguous(), *ctx.saved_tensors)
        out_feat, in_feat = grad_weight.size()[1:]
        print("grad_weight_column_major", grad_weight.flatten())
        grad_weight_row_major = grad_weight.view(-1, in_feat, out_feat).transpose(-1, -2).contiguous().view(-1, out_feat, in_feat)
        return grad_inp, None, grad_weight_row_major


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
            linear = nn.Linear(in_features=self.in_feat, out_features=self.out_feat)
            self.weight.data[i] = linear.weight.data

    def forward(self, inp, gate):
        return MOEFunction.apply(inp, gate, self.weight)


class MOELayer_raw(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=4096):
        super(MOELayer_raw, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Parameter(
            torch.Tensor(num_expert, out_feat, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, out_features=self.out_feat)
            self.weight.data[i] = linear.weight.data
    
    def forward(self, inp, gate):
        gate_long = gate.long()
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.out_feat))
        for i in range(batch_size):
            x[i] = self.weight[gate_long[i]] @ inp[i]
        return x


def test():
    batch_size = 4
    num_expert = 4
    in_feat = 2
    out_feat = 3

    moe = MOELayer(num_expert, in_feat, out_feat).cuda()
    moe_raw = MOELayer_raw(num_expert, in_feat, out_feat).cuda()
    moe_raw.weight.data = moe.weight.data.clone()

    inp = torch.rand(batch_size, in_feat).cuda()
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda()

    output = moe(inp, gate)
    output_raw= moe_raw(inp.clone(), gate.clone())

    print(output)
    print(output_raw)

    y = output.mean()
    y.backward()

    y_raw = output_raw.mean()
    y_raw.backward()

    print(moe.weight.grad)
    print(moe_raw.weight.grad)


if __name__ == '__main__':
    test()
