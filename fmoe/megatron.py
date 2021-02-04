from .layers import FMoETransformerMLP
from .distributed import DistributedGroupedDataParallel

def create_moe_mlp(args, model_parallel_rank, group):
    assert (
        args.seq_length * args.batch_size % args.model_parallel_size == 0
    ), "Batch size x sequence length should be multiple of mp size"
    if not args.distributed_experts:
        world_size = 1
    else:
        world_size = args.world_size
    fmoe = FMoETransformerMLP(
        args.num_experts,
        d_model=args.hidden_size,
        d_hidden=args.hidden_size * 4,
        world_size=world_size,
        model_parallel_size=args.model_parallel_size,
        model_parallel_rank=model_parallel_rank,
        mp_group=group,
    )
    return fmoe


def fmoefy(model, num_experts=None, distributed_experts=True):
    from megatron import get_args
    from megatron import mpu
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
        l.mlp = create_moe_mlp(args,
                mpu.get_model_parallel_rank(),
                mpu.get_model_parallel_group())
    return model


class DistributedDataParallel(DistributedGroupedDataParallel):
    def __init__(self, module):
        from megatron import mpu
        super(DistributedDataParallel, self).__init__(
            module,
            mp_group=mpu.get_model_parallel_group(),
            dp_group=mpu.get_data_parallel_group()
        )
