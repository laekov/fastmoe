r"""
distributed support for Megatron
"""
from fmoe.distributed import DistributedGroupedDataParallel


class DistributedDataParallel(DistributedGroupedDataParallel):
    r"""
    A wrapper that is used to replace the DDP module provided by Megatron, which
    is adapted to enable the sophiscated parallel and reduction strategies in
    Fast MoE.
    """

    def __init__(self, module):
        from megatron import mpu

        super().__init__(
            module,
            mp_group=mpu.get_model_parallel_group(),
            dp_group=mpu.get_data_parallel_group(),
        )

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
