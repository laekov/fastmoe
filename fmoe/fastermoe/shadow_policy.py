import os
import torch
import torch.distributed as dist


from .config import float_from_env, switch_from_env
from fmoe.functions import get_moe_group


def global_policy(local_expert_count, _gec, num_expert, world_size):
    r"""
    This is the policy for two-layer MLPs, using the formula in the PPoPP paper.
    A few parameters are used in this policy.
    * `d_model`: feature length of the MLP input and output.
    * `alpha`: the ratio of the MLP's hidden size to `d_model`.
    * `bw_net`: bandwidth of the network (GBps)
    * `bw_mm`: computation throughput of performing GeMM (FLOPs)
    """
    bw_net = float_from_env('FMOE_FASTER_GLBPLC_NETBW', 50 * 1e9 / 8)
    bw_mm = float_from_env('FMOE_FASTER_GLBPLC_GPUTP', 11.5e12)
    alpha = float_from_env('FMOE_FASTER_GLBPLC_ALPHA', 2)
    d_model = float_from_env('FMOE_FASTER_GLBPLC_DMODEL', 2048)

    moe_group = get_moe_group()
    local_expert_count = local_expert_count.cuda()
    agecs = [torch.empty_like(local_expert_count) for _ in range(world_size)]
    dist.all_gather(agecs, local_expert_count, group=moe_group)
    all_global_expert_count = torch.stack(agecs)

    # TODO: data type other than float
    data_size = 4 

    fwd_expert_counts = all_global_expert_count.sum(1).cpu()
    B_ws, indices = fwd_expert_counts.flatten().sort(0, descending=True)

    alphaH2 = alpha * (d_model ** 2)
    B_w = B_ws[0]

    comm = float('+inf')
    send_feature_time = d_model * data_size / bw_net
    send_model_time = 2 * alphaH2 * data_size / bw_net
    comp_time = 4 * alphaH2 / bw_mm
    lat_base = 3 * comp_time * B_w + 4 * send_feature_time * B_w

    res = torch.zeros(world_size * num_expert, dtype=torch.bool)
    shadow_time = 0

    for i, index in enumerate(indices):
        if i + 1 == indices.numel():
            break
        B_k = B_ws[i + 1]
        shadow_time += send_model_time
        lat_new = 3 * comp_time * B_k + 4 * send_feature_time * B_k + shadow_time

        if lat_new < lat_base:
            lat_base = lat_new
            res[index] = True
        else:
            break
    return res


def no_shadow_policy(_lec, _gec, num_expert, world_size):
    res = torch.zeros(world_size * num_expert, dtype=bool)
    return res


def get_shadow_policy(d_model=None):
    if d_model is not None and 'FMOE_FASTER_GLBPLC_DMODEL' not in os.environ:
        os.environ['FMOE_FASTER_GLBPLC_DMODEL'] = str(d_model)
    if not switch_from_env('FMOE_FASTER_SHADOW_ENABLE'):
        return no_shadow_policy
    return global_policy
