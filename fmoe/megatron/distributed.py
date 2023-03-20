r"""
distributed support for Megatron
"""
import torch

from fmoe.distributed import DistributedGroupedDataParallel


_groups = None


def _set_groups(**kwargs):
    global _groups
    _groups = kwargs


def get_moe_group():
    return _groups["moe_group"]


def _init():
    from megatron import get_args
    from megatron import mpu
    args = get_args()

    # Create a comm prependicular to the pipeline group as gate group
    stage_size = args.world_size // args.pipeline_model_parallel_size
    for i in range(0, args.world_size, stage_size):
        ranks = range(i, i + stage_size)
        group = torch.distributed.new_group(ranks)
        if args.rank in ranks:
            gate_group = group

    _set_groups(
            dp_group=mpu.get_data_parallel_group(),
            moe_group=mpu.get_data_parallel_group(),
            gate_group=gate_group)


class DistributedDataParallel(DistributedGroupedDataParallel):
    r"""
    A wrapper that is used to replace the DDP module provided by Megatron, which
    is adapted to enable the sophiscated parallel and reduction strategies in
    Fast MoE.
    """
    
    def __init__(self, module, accumulate_allreduce_grads_in_fp32=False, use_contiguous_buffers_in_ddp=False):
        assert not accumulate_allreduce_grads_in_fp32, "FastMoE not supports accumulate_allrecude_grads_in_fp32"
        assert not use_contiguous_buffers_in_ddp, "FastMoE not supports use_contiguous_buffers_in_ddp"
        
        if _groups is None:
            _init()
        super().__init__(module, **_groups)

    def set_input_tensor(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.set_input_tensor(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.state_dict(*args, **kwargs)

    def state_dict_for_save_checkpoint(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.state_dict_for_save_checkpoint(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        r"""
        Keep consitency with Megatron
        """
        return self.module.load_state_dict(*args, **kwargs)
