import torch.distributed as dist


def get_torch_default_comm():
    try:
        comm = dist.distributed_c10d._get_default_group()
        return comm
    except Exception as e:
        print('Error {}'.format(e))
        pass
    try:
        comm = dist.distributed_c10d._default_pg
        if comm is not None:
            return comm
    except Exception as _:
        pass
    raise RuntimeError('Unsupported PyTorch version')
    return None


