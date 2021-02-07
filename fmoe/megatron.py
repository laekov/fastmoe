r'''
The adaptor to seamlessly enable FastMoE in Megatron-LM v2.0 with at most two
lines of modification.
See `exapmles/megatron` for usage instructions.
'''
import torch

from .transformer import FMoETransformerMLP
from .distributed import DistributedGroupedDataParallel
from .utils import get_torch_default_comm


class MegatronMLP(FMoETransformerMLP):
    r'''
    Make the FMoETransformerMLP layer that distributes experts across
    communication group `group` to replace the original MLP layer in Megatron.
    '''
    def __init__(self, args, group):
        assert (args.seq_length * args.micro_batch_size
                % args.tensor_model_parallel_size == 0
        ), "Batch size x sequence length should be multiple of mp size"
        if not args.distributed_experts:
            world_size = 1
        else:
            world_size = args.world_size
        super().__init__(args.num_experts,
                d_model=args.hidden_size, d_hidden=args.hidden_size * 4,
                world_size=world_size, mp_group=group)
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(args.hidden_size, dtype=torch.float32)
        )

    def forward(self, inp):
        return super().forward(inp), self.bias


def fmoefy(model, num_experts=None, distributed_experts=True):
    r'''
    Replace MLP layers in a transformer-based model in Megatron by MoE.
    * `model` should be a standard Megatron model that has
    `model.language_model.transformer.layers` as transformer layers, which is an
    array of transformer blocks that contain an `mlp` member.
    * `distributed_expert` is set to True if different experts are located in
    different workers. Otherwise, the experts on the workers are identical, and
    they are trained in data-parallel mode. This can be useful when testing on
    small models that do not require high training throughput or large parameter
    capacity.
    Note that pipeline parallel is not supported yet. When distributed experts
    are enabled, their communicator should be Megatron's
    tensor_model_parall_comm x data_parallel_comm, which is not created.
    '''
    from megatron import get_args
    args = get_args()
    if num_experts is not None:
        args.num_experts = num_experts
    assert (
        'num_experts' in args
    ), 'num_experts should be specified in arguments or fmoefy function'

    # Set distributed_experts to None to use default setting in args
    if distributed_experts is not None:
        args.distributed_experts = distributed_experts

    for l in model.language_model.transformer.layers:
        l.mlp = MegatronMLP(args, get_torch_default_comm())
    return model


class DistributedDataParallel(DistributedGroupedDataParallel):
    r'''
    A wrapper that is used to replace the DDP module provided by Megatron, which
    is adapted to enable the sophiscated parallel and reduction strategies in
    Fast MoE.
    '''
    def __init__(self, module):
        from megatron import mpu
        super().__init__(
            module,
            mp_group=mpu.get_model_parallel_group(),
            dp_group=mpu.get_data_parallel_group()
        )

    def state_dict(self, *args, **kwargs):
        r'''
        Keep consitency with Megatron
        '''
        return self.module.state_dict(*args, **kwargs)

    def state_dict_for_save_checkpoint(self, *args, **kwargs):
        r'''
        Keep consitency with Megatron
        '''
        return self.module.state_dict_for_save_checkpoint(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        r'''
        Keep consitency with Megatron
        '''
        return self.module.load_state_dict(*args, **kwargs)
