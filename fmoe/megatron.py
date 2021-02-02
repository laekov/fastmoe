from .layers import FMoETransformerMLP


def create_moe_mlp(args, model_parallel_rank, group):
    assert (
        args.seq_length * args.batch_size % args.model_parallel_size == 0
    ), "Num experts should be multiple of mp size"
    num_experts = args.num_experts // args.model_parallel_size
    fmoe = FMoETransformerMLP(
        num_experts,
        d_model=args.hidden_size,
        d_hidden=args.hidden_size * 4,
        world_size=args.world_size,
        model_parallel_size=args.model_parallel_size,
        model_parallel_rank=model_parallel_rank,
        group=group,
    )
    return fmoe
