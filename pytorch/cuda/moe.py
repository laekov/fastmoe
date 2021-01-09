import math
from torch import nn
from torch.autograd import Function
import torch

import moe_cuda


class MOEFunction(Function):
    @staticmethod
    def forward(ctx, inp, gate, weight):
        # out_feat, in_feat = weight.size()[1:]
        # weight_column_major = weight.transpose(-1, -2).contiguous().view(-1, out_feat, in_feat)
        expert_count, pos = moe_cuda.expert_count(weight, gate)
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


class MOELayer(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024):
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
        return MOEFunction.apply(inp, gate.int(), self.weight)


class MOELayer_raw(nn.Module):
    def __init__(self, num_expert=32, in_feat=1024, out_feat=1024):
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
            # print(linear.weight.shape)
            self.weight.data[i] = linear.weight.data
    
    def forward(self, inp, gate):
        gate_long = gate.long()
        batch_size = inp.size(0)
        x = inp.new_zeros((batch_size, self.out_feat))
        for i in range(batch_size):
            x[i] = inp[i] @ self.weight[gate_long[i]].t()
        return x


def test_module(moe, linear, inp, gate):
    linear.zero_grad()
    moe.zero_grad()
    x = linear(inp)
    output = moe(x, gate)
    y = output.mean()
    y.backward()
    return output, moe.weight.grad, linear.weight.grad, linear.bias.grad


def test():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    batch_size = 4
    num_expert = 2
    in_feat = 6
    out_feat = 7

    linear = nn.Linear(in_feat, in_feat).cuda()

    moe = MOELayer(num_expert, in_feat, out_feat).cuda()
    moe_raw = MOELayer_raw(num_expert, in_feat, out_feat).cuda()
    moe_raw.weight.data = moe.weight.data.clone()

    inp = torch.rand(batch_size, in_feat).cuda()
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda()

    moe_out = test_module(moe, linear, inp.clone(), gate.clone())
    raw_out = test_module(moe_raw, linear, inp.clone(), gate.clone())

    names = ['Out', 'Moe wei', 'Linear wei', 'Linear bias']
    names = ['Out']
    for name, mo, ro in zip(names, moe_out, raw_out):
        err = (mo - ro).abs().sum()
        print('{} abs err {}'.format(name, err))

def test_dp():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    batch_size = 6
    num_expert = 4
    in_feat = 2
    out_feat = 3

    inp = torch.rand(batch_size, in_feat).cuda()
    gate = torch.randint(low=0, high=num_expert, size=(batch_size, ), requires_grad=False).int().cuda()

    print("data parallel of a nn.Linear model")
    linear = nn.Linear(in_feat, in_feat).cuda()
    linear_dp = torch.nn.DataParallel(linear, device_ids=[0,1,2])
    output = linear_dp(inp)
    print("successful!")

    print("data parallel of our MoE model")
    moe = MOELayer(num_expert, in_feat, out_feat).cuda()
    moe_dp = torch.nn.DataParallel(moe, device_ids=[0,1,2])
    for i in range(5):
        output = moe_dp(inp, gate)



if __name__ == '__main__':
    test()
    # test_dp()
