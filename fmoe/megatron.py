r"""
The adaptor to seamlessly enable FastMoE in Megatron-LM v2.0 with at most two
lines of modification.
See `examples/megatron` for usage instructions.
"""
import os
import math
import numpy as np
import random
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import FMoETransformerMLP
from .distributed import DistributedGroupedDataParallel
from .balance import update_balance_profile, reset_balance_profile
from .utils import get_torch_default_comm


class _FakeMegatronMLP(nn.Module):
    r"""
    A fake mlp without model parallelism for correctness testing
    """

    def __init__(self, args, _):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_hidden_size)
        self.fc2 = nn.Linear(args.hidden_hidden_size, args.hidden_size)

    def forward(self, x):
        r"""
        Directly use GeLU
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x, torch.zeros_like(x)


def _megatron_init_method(self, rng, sigma):
    r"""
    Init method based on N(0, sigma).
    Copied from Megatron-LM
    """
    device = self.weight.device
    dtype = self.weight.dtype
    weight = rng.normal(loc=0.0, scale=sigma, size=tuple(self.weight.size()))
    self.weight.data = torch.from_numpy(weight).to(dtype=dtype, device=device)

    if self.bias is not None:
        # Always initialize bias to zero.
        with torch.no_grad():
            self.bias.zero_()


def _random_init_weight(self, rng):
    r"""
    Copied from torch.nn.init.kaiming_uniform_
    """
    fan = nn.init._calculate_correct_fan(self.weight[0], "fan_in")
    gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    device = self.weight.device
    dtype = self.weight.dtype
    weight = rng.uniform(-bound, bound, size=tuple(self.weight.size()))
    self.weight.data = torch.from_numpy(weight).to(dtype=dtype, device=device)

    if self.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in)
        bias = rng.uniform(-bound, bound, size=tuple(self.bias.size()))
        self.bias.data = torch.from_numpy(bias).to(dtype=dtype, device=device)


balance_dict = {}
num_layers = 0


def reset_gate_hook():
    from megatron import get_args

    global balance_dict, num_layers
    reset_balance_profile(balance_dict, num_layers, get_args().balance_strategy)


def get_balance_profile():
    global balance_dict
    return balance_dict


def generate_megatron_gate_hook(layer_idx, num_expert_global):
    from megatron import get_args

    balance_strategy = get_args().balance_strategy

    def megatron_gate_hook(gate_top_k_idx, gate_score_top_k, gate_state_dict):
        global balance_dict
        update_balance_profile(
            balance_dict,
            gate_top_k_idx,
            gate_score_top_k,
            gate_state_dict,
            layer_idx,
            num_expert_global,
            balance_strategy,
        )

    return megatron_gate_hook


def add_fmoe_args(parser):
    group = parser.add_argument_group(title="fastmoe")

    group.add_argument("--fmoefy", action="store_true")
    group.add_argument("--num-experts", type=int, default=None)
    group.add_argument("--top-k", type=int, default=2)
    group.add_argument("--balance-loss-weight", type=float, default=1)
    group.add_argument("--balance-strategy", type=str, default=None)

    return parser


def add_balance_log(writer, iteration):
    from megatron import is_last_rank

    balance_dict_tensor = torch.vstack(
        [torch.tensor(item, device=item[0].device) for item in balance_dict.values()]
    ).detach()
    world_group = get_torch_default_comm()
    world_size = torch.distributed.get_world_size(group=world_group)
    torch.distributed.all_reduce(balance_dict_tensor, group=world_group)
    balance_dict_tensor /= world_size

    if writer and is_last_rank():
        for idx, metric_name in enumerate(balance_dict):
            for layer_id, val in enumerate(balance_dict_tensor[idx]):
                writer.add_scalar(
                    f"balance-{metric_name}/layer-{layer_id}", val.item(), iteration
                )
            writer.add_scalar(
                f"balance-{metric_name}/all",
                balance_dict_tensor[idx].mean().item(),
                iteration,
            )

    reset_gate_hook()


def patch_forward_step(forward_step_func):
    r"""
    Patch model's forward_step_func to support balance loss
    """

    from megatron.mpu import is_pipeline_last_stage
    from megatron import get_args

    if not get_args().balance_strategy:
        return forward_step_func

    def forward_step_with_balance_loss(data_iterator, model, input_tensor):
        args = get_args()
        output = forward_step_func(data_iterator, model, input_tensor)

        if is_pipeline_last_stage():
            loss_name = args.balance_strategy + "_loss"

            (loss, state_dict), bal_loss = (
                output,
                (
                    torch.tensor(
                        balance_dict[loss_name],
                        device=balance_dict[loss_name][0].device,
                    ).mean()
                    * args.balance_loss_weight
                ).float(),
            )

            # avarage across world group
            world_group = get_torch_default_comm()
            world_size = torch.distributed.get_world_size(group=world_group)
            averaged_bal_loss = bal_loss.clone().detach()
            torch.distributed.all_reduce(averaged_bal_loss, group=world_group)
            averaged_bal_loss /= world_size

            loss += bal_loss
            state_dict[loss_name] = averaged_bal_loss

            return loss, state_dict
        else:
            return output

    return forward_step_with_balance_loss


def patch_model_provider(model_provider):
    from megatron import get_args

    def fmoefied_model_provider():
        args = get_args()
        return fmoefy(
            model_provider(),
            num_experts=args.num_experts,
            hidden_hidden_size=4 * args.hidden_size // args.top_k,
            top_k=args.top_k,
        )

    return fmoefied_model_provider


class MegatronMLP(FMoETransformerMLP):
    r"""
    Make the FMoETransformerMLP layer that distributes experts across
    communication group `group` to replace the original MLP layer in Megatron.
    """

    def __init__(self, args, group, layer_idx):
        assert (
            args.seq_length * args.micro_batch_size % args.tensor_model_parallel_size
            == 0
        ), "Batch size x sequence length should be multiple of mp size"
        if not args.distributed_experts:
            world_size = 1
        else:
            world_size = args.world_size
        gate = None
        if not args.balance_strategy or args.balance_strategy == "gshard":
            from .gates import NaiveGate

            gate = NaiveGate
        elif args.balance_strategy == "noisy":
            from .gates import NoisyGate

            gate = NoisyGate
        else:
            assert False, "Undefined balance strategy {}" % (args.balance_strategy)
        super().__init__(
            args.num_experts,
            top_k=args.top_k,
            d_model=args.hidden_size,
            d_hidden=args.hidden_hidden_size,
            world_size=world_size,
            mp_group=group,
            expert_dp_comm="none" if args.distributed_experts else "dp",
            gate_hook=generate_megatron_gate_hook(
                layer_idx, args.num_experts * world_size
            ),
            gate=gate,
        )
        self.hidden_size = args.hidden_size
        if args.distributed_experts:
            self.rank = args.rank
        else:
            self.rank = 0
        self.sigma = args.init_method_std
        self.num_layers = args.num_layers
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Initialize the weight as linear layers.
        As megatron is using fixed random seed for some nasty stuff, an
        additional numpy rng is used.
        """
        rng = np.random.default_rng(np.random.randint(2048) + self.rank)
        _megatron_init_method(self.experts.htoh4, rng, self.sigma)
        std = self.sigma / math.sqrt(2.0 * self.num_layers)
        _megatron_init_method(self.experts.h4toh, rng, std)

    def forward(self, inp):
        return (
            super().forward(inp),
            torch.zeros(self.hidden_size, dtype=inp.dtype, device=inp.device),
        )


def fmoefy(
    model,
    num_experts=None,
    distributed_experts=True,
    hidden_hidden_size=None,
    top_k=None,
):
    r"""
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
    """
    from megatron import get_args
    from megatron import mpu

    args = get_args()
    if num_experts is not None:
        args.num_experts = num_experts
    assert (
        "num_experts" in args
    ), "num_experts should be specified in arguments or fmoefy function"

    if hidden_hidden_size is not None:
        args.hidden_hidden_size = hidden_hidden_size
    elif not hasattr(args, "hidden_hidden_size"):
        args.hidden_hidden_size = args.hidden_size * 4

    if top_k is not None:
        args.top_k = top_k
    elif not hasattr(args, "top_k"):
        args.top_k = 2

    # Set distributed_experts to None to use default setting in args
    if distributed_experts is not None:
        args.distributed_experts = distributed_experts

    for idx, l in enumerate(model.language_model.transformer.layers):
        l.mlp = MegatronMLP(args, mpu.get_model_parallel_group(), idx)

    # initialize gate hook
    global num_layers, balance_dict
    num_layers = len(model.language_model.transformer.layers)
    reset_gate_hook()

    return model


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

def get_checkpoint_name(checkpoints_path, iteration,
                        release=False):
    """A unified checkpoint name."""
    from megatron import mpu

    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)
    # Use both the tensor and pipeline MP rank.
    if mpu.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(checkpoints_path, directory,
                            'mp_rank_{:02d}_dp_rank_{:04d}'.format(
                                mpu.get_tensor_model_parallel_rank(),
                                mpu.get_data_parallel_rank()
                                ),
                            'model_optim_rng.pt')
    return os.path.join(checkpoints_path, directory,
                        'mp_rank_{:02d}_{:03d}_dp_rank_{:04d}'.format(
                            mpu.get_tensor_model_parallel_rank(),
                            mpu.get_pipeline_model_parallel_rank(),
                            mpu.get_data_parallel_rank()
                            ),
                        'model_optim_rng.pt')

def save_checkpoint(iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint with expert parallel """
    from megatron import get_args
    from megatron import mpu

    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    if isinstance(model, DistributedDataParallel):
        model = model.module

    if torch.distributed.get_rank() == 0:
        print('saving checkpoint at iteration {:7d} to {}'.format(
            iteration, args.save), flush=True)

    data_parallel_rank = mpu.get_data_parallel_rank()

    # Arguments, iteration, and model.
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = iteration
    keep_vars = False if mpu.get_data_parallel_rank() == 0 else True
    state_dict['model'] = model.state_dict_for_save_checkpoint(keep_vars=keep_vars)

    if mpu.get_data_parallel_rank() != 0:

        def extract_expert_param(state_dict, expert_dp_comm='none'):
            state_dict_new = state_dict.__class__()
            for k, v in state_dict.items():
                # megatron uses both dict and OrderedDict in its state_dict
                if isinstance(v, OrderedDict) or isinstance(v, dict):
                    v_new = extract_expert_param(v, expert_dp_comm)
                    if len(v_new):
                        state_dict_new[k] = v_new
                elif hasattr(v, 'dp_comm') and v.dp_comm == expert_dp_comm:
                    state_dict_new[k] = v.detach()
            return state_dict_new

        state_dict['model'] = extract_expert_param(state_dict['model'], 'none') 

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()

    # RNG states.
    if not args.no_save_rng:
        state_dict['random_rng_state'] = random.getstate()
        state_dict['np_rng_state'] = np.random.get_state()
        state_dict['torch_rng_state'] = torch.get_rng_state()
        state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
        state_dict['rng_tracker_states'] \
            = mpu.get_cuda_rng_tracker().get_states()

    # Save.
    checkpoint_name = get_checkpoint_name(args.save, iteration)
    from megatron.checkpointing import ensure_directory_exists
    from megatron.checkpointing import get_checkpoint_tracker_filename
    ensure_directory_exists(checkpoint_name)
    torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('  successfully saved checkpoint at iteration {:7d} to {}'.format(
            iteration, args.save), flush=True)
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()
