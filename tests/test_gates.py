import os
import torch
import torch.distributed as dist
from fmoe.gates import GShardGate


def test_gshard_gate(d_model, batch_size, n_expert):
    gate = GShardGate(d_model, n_expert, dist.get_world_size()).cuda()
    x = torch.rand(batch_size, d_model).cuda()
    topk_idx, topk_val = gate(x)
    print('rank {} idx {}'.format(dist.get_rank(), topk_idx))
    print('rank {} val {}'.format(dist.get_rank(), topk_val))


if __name__ == '__main__':
    os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
    torch.distributed.init_process_group(backend="nccl")
    test_gshard_gate(4096, 1024, 4)
