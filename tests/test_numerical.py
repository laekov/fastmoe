from fmoe.layers import FMoE
from fmoe.transformer import _Expert
from fmoe.gates import NaiveGate

from moe import BruteForceMoELinear
import torch
import sys
import os

rank = 0
world_size = 1


def test_fmoe_linear():
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    batch_size = 4
    num_expert = 2
    d_model = 6
    d_hidden = 8
    top_k = 2
    activation = torch.nn.functional.gelu

    experts = _Expert(num_expert, d_model, d_hidden, activation).cuda()

    def expert_fn(inp, gate):
        return experts(inp, gate)

    moe = FMoE(
        num_expert=num_expert,
        d_model=d_model,
        gate=NaiveGate,
        world_size=world_size,
        mp_group=None,
        expert_fn=expert_fn,
        top_k=top_k,
    ).cuda()

    moe_raw = BruteForceMoELinear(
        activation=activation,
        num_expert=num_expert,
        d_model=d_model,
        world_size=world_size,
    ).cuda()

    if world_size == 1:
        moe_raw.weight_htoh4.data = experts.htoh4.weight.data.clone()
        moe_raw.weight_h4toh.data = experts.h4toh.weight.data.clone()
    else:
        weight_htoh4_array = [
            torch.empty_like(experts.htoh4.weight.data) for _ in range(world_size)
        ]
        torch.distributed.all_gather(weight_htoh4_array, experts.htoh4.weight.data)
        moe_raw.weight_htoh4.data = torch.cat(weight_htoh4_array, dim=0)

        weight_h4toh_array = [
            torch.empty_like(experts.h4toh.weight.data) for _ in range(world_size)
        ]
        torch.distributed.all_gather(weight_h4toh_array, experts.h4toh.weight.data)
        moe_raw.weight_h4toh.data = torch.cat(weight_h4toh_array, dim=0)

    inp = torch.rand(batch_size, d_model).cuda()

    gate_idx, gate_score = moe.gate(inp)
    print(gate_idx.shape, gate_score.shape)
    inp_repeated = inp.repeat_interleave(repeats=top_k, dim=0)

    moe_out = moe(inp).mean()
    raw_out = moe_raw(inp_repeated, gate_idx, gate_score).mean()

    moe_out.backward()
    raw_out.backward()

    moe_out = moe_out, experts.htoh4.weight.grad, experts.h4toh.weight.grad
    raw_out = raw_out, moe_raw.weight_htoh4.grad, moe_raw.weight_h4toh.grad

    names = ["output", "htoh4 weight grad", "h4toh weight grad"]
    if world_size > 1:
        ou, htoh4_grad, h4toh_grad = raw_out
        torch.distributed.all_reduce(htoh4_grad)
        torch.distributed.all_reduce(h4toh_grad)
        htoh4_grad = htoh4_grad[rank * num_expert : (rank + 1) * num_expert]
        h4toh_grad = h4toh_grad[rank * num_expert : (rank + 1) * num_expert]
        raw_out = ou, htoh4_grad, h4toh_grad
    for name, mo, ro in zip(names, moe_out, raw_out):
        err = (mo - ro).abs().sum()
        print("Rank {} {} abs err {}".format(rank, name, err))
        if err > 1e-3:
            sys.stderr.write("=========== moe out ==============\n")
            sys.stderr.write("{}\n".format(mo))
            sys.stderr.write("=========== raw out ==============\n")
            sys.stderr.write("{}\n".format(ro))
            assert False
    torch.cuda.synchronize()


def test_fmoe_linear_distributed():
    import subprocess
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "36666"
    ps, n = [], 2
    os.environ["WORLD_SIZE"] = str(n)

    for i in range(n):
        os.environ["RANK"] = str(i)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        p = subprocess.Popen([sys.executable, __file__], stdout=subprocess.PIPE)
        ps.append(p)

    for p in ps:
        p.wait()
        retc = p.poll()
        assert retc == 0


if __name__ == "__main__":
    # os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
    # os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
    if int(os.environ["WORLD_SIZE"]) > 1:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    test_fmoe_linear()
