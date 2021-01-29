from torch import nn
from .moe import FMoE
from .moe_function import moe
from .fmoe import FMoETransformerMLP


class FFFN(nn.Module):
    def __init__(self, num_expert=32, d_model=1024, d_hidden=4096, 
            world_size=None, activation=torch.nn.functional.gelu,
            top_k=2, pre_lnorm=False):
        super(FFFN, self).__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.world_size = world_size
        self.activation = activation
        self.top_k = top_k
        self.pre_lnorm = pre_lnorm

        self.htoh4 = FMoE(num_expert, d_model, d_hidden,
                world_size=world_size)
        self.h4toh = FMoE(num_expert, d_hidden, d_model, 
                world_size=world_size)
        self.gate = nn.Linear(d_model, num_expert * world_size)
        self.layer_norm = nn.LayerNorm(d_model)
        self.bias = torch.nn.parameter.Parameter(torch.zeros(d_model,
                dtype=torch.float32)) 

    def forward(self, inp):
        # import pdb; pdb.set_trace()
        residual = inp
        if self.pre_lnorm:
            inp = self.layer_norm(inp)

        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(gate, k=self.top_k, dim=-1,
                largest=True, sorted=False) # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k 
        gate_score = F.softmax(gate_top_k_val, dim=-1).unsqueeze(1) 
        gate_top_k_idx = gate_top_k_idx.view(-1) # (BxLxtop_k)

        inp = inp.view(-1, self.d_model).repeat_interleave(repeats=self.top_k, 
                dim=0) # (BxLxtop_k) x d_model
        x = self.htoh4(inp, gate_top_k_idx)
        x = self.activation(x)
        x = self.h4toh(x, gate_top_k_idx)

        core_out = x.view(-1, self.top_k, self.d_model) # (BxL) x top_k x d_model 
        core_out = torch.bmm(gate_score, core_out) # (BxL) x 1 x d_model
        core_out = core_out.view(residual.size(0), residual.size(1), self.d_model)
        output = core_out + residual

        if not self.pre_lnorm:
            output = self.layer_norm(output)
        return output, self.bias


def create_moe_mlp(args):
    assert args.num_experts % args.model_parallel_size == 0, 'Num experts should be multiple of mp size'
    num_experts = args.num_experts // args.model_parallel_size 
    fmoe = FMoETransformerMLP(num_experts, 
            d_model=args.hidden_size, 
            d_hidden=args.hidden_size * 4,
            world_size=args.model_parallel_size)
    return fmoe
    
