r"""
Utils to play with PyTorch.
"""
import torch
import torch.distributed as dist


# pylint: disable=broad-except
# pylint: disable=protected-access
def get_torch_default_comm():
    r"""
    The NCCL communicator is needed so that Fast MoE can perform customized
    communication operators in the C code. However, it is not a publicly
    available variable. Therefore, a hacking class of the `ProcessGroupNCCL`
    in Fast MoE's C code takes the `_default_pg` and tries to dig the
    communicator out from the object. As PyTorch's private interface varies from
    time to time, different hacking techniques are tried one-by-one to be
    compatible with various versions of PyTorch.
    """
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as _:
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError("Unsupported PyTorch version")


def get_rank_0_in_comm(comm):
    world_size = dist.get_world_size(comm)
    x = torch.tensor([dist.get_rank()], dtype=torch.int64, device='cuda')
    ys = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(ys, x, group=comm)
    root_rank = ys[0].item()
    return root_rank

