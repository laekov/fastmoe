import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DistributedGroupedDataParallel(nn.Module):
    def __init__(self, module, mp_group=None, dp_group=None, world_group=None,
            auto_allreduce=False):
        assert not auto_allreduce, 'Automatic all-reduce is not implemented yet'

        super(DistributedGroupedDataParallel, self).__init__()
        self.module = module

        self.comms = dict()
        if mp_group is not None:
            self.comms['mp'] = mp_group
        if dp_group is not None:
            self.comms['dp'] = dp_group
        else:
            self.comms['dp'] = torch.distributed.distributed_c10d._get_default_group()
        if world_group is None:
            self.comms['world'] = torch.distributed.distributed_c10d._get_default_group()
        else:
            self.comms['world'] = world_group

        def allreduce_params(no_scale=False, reduce_after=False, 
                fp32_allreduce=False):
            groups = dict()
            for p in self.module.parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                if hasattr(p, 'dp_comm'):
                    dp_comm = p.dp_comm
                else:
                    dp_comm = 'dp'
                group_key = (dp_comm, p.dtype)
                if group_key not in groups:
                    groups[group_key] = [p]
                else:
                    groups[group_key].append(p)
            for dp_comm, dtype in groups:
                if dp_comm not in self.comms:
                    continue
                group = groups[dp_comm, dtype]
                comm = self.comms[dp_comm]
                grads = [p.grad.data for p in group]
                coalesced = _flatten_dense_tensors(grads)
                if fp32_allreduce and dtype != torch.float32:
                    coalesced = coalesced.float()
                if not no_scale and not reduce_after:
                    coalesced /= comm.size()
                torch.distributed.all_reduce(coalesced, group=comm)
                torch.cuda.synchronize()
                if not no_scale and reduce_after:
                    coalesced /= comm.size()
                synced = _unflatten_dense_tensors(coalesced, grads)
                for g, s in zip(grads, synced):
                    g.copy_(s)

        self.allreduce_params = allreduce_params

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

