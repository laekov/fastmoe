import math
from torch import nn
from torch.autograd import Function
import torch

import moe_cuda

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class MOEFunction(Function):
    @staticmethod
    def forward(ctx, inp, gate, weight1, weight2):
        # out_feat, in_feat = weight.size()[1:]
        # weight_column_major = weight.transpose(-1, -2).contiguous().view(-1, out_feat, in_feat)
        output = moe_cuda.forward(inp, gate, weight1, weight2)
        variables = [inp, gate, weight1, weight2]
        ctx.save_for_backward(*variables)

        return output[0]

    @staticmethod
    def backward(ctx, grad_out):
        # print("grad_out", grad_out)
        # print("input", ctx.saved_tensors[0])
        grad_inp, grad_weight = moe_cuda.backward(
            grad_out.contiguous(), *ctx.saved_tensors)
        out_feat, in_feat = grad_weight.size()[1:]
        # print("grad_weight_column_major", grad_weight.flatten())
        grad_weight_row_major = grad_weight.view(-1, in_feat, out_feat).transpose(-1, -2).contiguous().view(-1, out_feat, in_feat)
        return grad_inp, None, grad_weight_row_major


class MOELayer(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, hidden_feat=4096, out_feat=1024):
        super(MOELayer, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.hidden_feat = hidden_feat
        self.out_feat = out_feat
        self.weight1 = nn.Parameter(
            torch.Tensor(num_expert, hidden_feat, in_feat))
        self.weight2 = nn.Parameter(
            torch.Tensor(num_expert, out_feat, hidden_feat))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, out_features=self.hidden_feat)
            self.weight1.data[i] = linear.weight.data
            linear = nn.Linear(in_features=self.hidden_feat, out_features=self.out_feat)
            self.weight2.data[i] = linear.weight.data

    def forward(self, inp, gate):
        return MOEFunction.apply(inp, gate, self.weight1, self.weight2)


class MOELayer_raw(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, hidden_feat=4096, out_feat=1024):
        super(MOELayer_raw, self).__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.hidden_feat = hidden_feat
        self.out_feat = out_feat
        self.weight1 = nn.Parameter(
            torch.Tensor(num_expert, hidden_feat, in_feat))
        self.weight2 = nn.Parameter(
            torch.Tensor(num_expert, out_feat, hidden_feat))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_expert):
            linear = nn.Linear(in_features=self.in_feat, out_features=self.hidden_feat)
            # print(linear.weight.shape)
            self.weight1.data[i] = linear.weight.data
            linear = nn.Linear(in_features=self.hidden_feat, out_features=self.out_feat)
            self.weight2.data[i] = linear.weight.data
    
    def forward(self, inp, gate):
        gate_long = gate.long()
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.out_feat))
        # print(self.weight2)
        for i in range(batch_size):
            hid = inp[i] @ self.weight1[gate_long[i]].t()
            # print(hid)
            x[i] = hid @ self.weight2[gate_long[i]].t()
        return x


def test_module(moe, linear, inp, gate):
    linear.zero_grad()
    moe.zero_grad()
    x = linear(inp)
    output = moe(x, gate)
    print(output)
    return output
    print(output)
    y = output.mean()
    y.backward()
    return output, moe.weight.grad, linear.weight.grad, linear.bias.grad


def test():
    batch_size = 4
    num_expert = 2
    in_feat = 6
    hidden_feat = 12
    out_feat = 7

    linear = nn.Linear(in_feat, in_feat).cuda()

    moe = MOELayer(num_expert, in_feat, hidden_feat, out_feat).cuda()
    moe_raw = MOELayer_raw(num_expert, in_feat, hidden_feat, out_feat).cuda()
    moe_raw.weight1.data = moe.weight1.data.clone()
    moe_raw.weight2.data = moe.weight2.data.clone()

    inp = torch.rand(batch_size, in_feat).cuda()
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda()

    moe_out = test_module(moe, linear, inp.clone(), gate.clone())
    raw_out = test_module(moe_raw, linear, inp.clone(), gate.clone())

    names = ['Out', 'Moe wei', 'Linear wei', 'Linear bias']
    names = ['Out']
    for name, mo, ro in zip(names, moe_out, raw_out):
        err = (mo - ro).abs().sum()
        print('{} abs err {}'.format(name, err))

if __name__ == '__main__':
    test()
