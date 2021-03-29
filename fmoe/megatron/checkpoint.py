r"""
Support for Megatron to enable saving parameters of different experts on
different ranks.
"""
import os
import sys
import random
from collections import OrderedDict
import numpy as np
import torch


def get_fmoe_checkpoint_name(
    checkpoints_path, iteration, release=False, data_parallel_rank=-1
):
    """A unified checkpoint name, allowing specifying a data parallel rank"""
    from megatron import mpu
    from megatron.checkpointing import get_checkpoint_name

    if data_parallel_rank == -1:
        data_parallel_rank = mpu.get_data_parallel_rank()
    if data_parallel_rank == 0:
        return get_checkpoint_name(checkpoints_path, iteration, release)

    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    # Use both the tensor and pipeline MP rank.
    if mpu.get_pipeline_model_parallel_world_size() == 1:
        return os.path.join(
            checkpoints_path,
            directory,
            "mp_rank_{:02d}_dp_rank_{:04d}".format(
                mpu.get_tensor_model_parallel_rank(), data_parallel_rank
            ),
            "model_optim_rng.pt",
        )
    return os.path.join(
        checkpoints_path,
        directory,
        "mp_rank_{:02d}_{:03d}_dp_rank_{:04d}".format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            data_parallel_rank,
        ),
        "model_optim_rng.pt",
    )


def save_checkpoint(iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint with expert parallel """
    # TODO: update patch
    from megatron import get_args
    from megatron import mpu
    from megatron import print_rank_last

    expert_dp_comm = "none"

    if mpu.get_data_parallel_rank() == 0:
        # at dp rank 0, we still follows the native load_checkpoint by megatron
        from megatron.checkpointing import save_checkpoint as save_checkpoint_native

        save_checkpoint_native(iteration, model, optimizer, lr_scheduler)
        return

    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    if hasattr(model, 'module'):
        model = model.module

    print_rank_last(
        "saving checkpoint at iteration {:7d} to {}".format(iteration, args.save)
    )

    # Arguments, iteration, and model.
    state_dict = {}
    state_dict["model"] = model.state_dict_for_save_checkpoint(
        keep_vars=(mpu.get_data_parallel_rank() > 0)
    )

    def extract_expert_param(state_dict, expert_dp_comm="none"):
        state_dict_new = state_dict.__class__()
        for k, v in state_dict.items():
            # megatron uses both dict and OrderedDict in its state_dict
            if isinstance(v, (OrderedDict, dict)):
                v_new = extract_expert_param(v, expert_dp_comm)
                if len(v_new) > 0:
                    state_dict_new[k] = v_new
            elif hasattr(v, "dp_comm") and v.dp_comm == expert_dp_comm:
                state_dict_new[k] = v.detach()
        return state_dict_new

    state_dict["model"] = extract_expert_param(state_dict["model"], expert_dp_comm)

    # Optimizer stuff.
    if not args.no_save_optim:
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
            param_global_idx = 0
            for param_group in optimizer.optimizer.param_groups:
                for param in param_group["params"]:
                    if not (
                        hasattr(param, "dp_comm") and param.dp_comm == expert_dp_comm
                    ):
                        # this parameter is not an expert parameter
                        # thus there is no need to save its state in current rank
                        # since it has been saved by data parallel rank 0
                        if args.fp16:
                            # fp16 optimizer may have empty state due to overflow
                            state_dict["optimizer"]["optimizer"]["state"].pop(
                                param_global_idx, None
                            )
                        else:
                            state_dict["optimizer"]["state"].pop(param_global_idx)
                    param_global_idx += 1
            if args.fp16:
                state_dict["optimizer"]["optimizer"].pop("param_groups")
                # fp32_from_fp16_params in state_dict is not a copy
                # but a reference to optimizer.fp32_from_fp16_params,
                # changing it in state_dict will change
                # optimizer.fp32_from_fp16_params as well
                # thus we create an empty fp32_from_fp16_params in state_dict
                # and only insert expert parameters.
                fp32_from_fp16_params = state_dict["optimizer"]["fp32_from_fp16_params"]
                state_dict["optimizer"]["fp32_from_fp16_params"] = []
                for param_group in fp32_from_fp16_params:
                    param_group_copy = []
                    for param in param_group:
                        param_copy = (
                            param
                            if hasattr(param, "dp_comm")
                            and param.dp_comm == expert_dp_comm
                            else None
                        )
                        param_group_copy.append(param_copy)
                    state_dict["optimizer"]["fp32_from_fp16_params"].append(
                        param_group_copy
                    )
            else:
                state_dict["optimizer"].pop("param_groups")

    # Save.
    checkpoint_name = get_fmoe_checkpoint_name(args.save, iteration)
    from megatron.checkpointing import ensure_directory_exists
    from megatron.checkpointing import get_checkpoint_tracker_filename

    ensure_directory_exists(checkpoint_name)
    torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            "  successfully saved checkpoint at iteration {:7d} to {}".format(
                iteration, args.save
            ),
            flush=True,
        )
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def merge_state_dict(state_dict_rank0, state_dict_local, fp16):
    """merge two state dicts, one from data parallel rank 0,
    another only contains expert states"""
    # from megatron import print_rank_last

    def merge_model(state_dict_rank0, state_dict_local):
        for k, v in state_dict_local.items():
            # megatron uses both dict and OrderedDict in its state_dict
            if isinstance(v, (OrderedDict, dict)):
                merge_model(state_dict_rank0[k], v)
            else:
                state_dict_rank0[k] = v

    merge_model(state_dict_rank0["model"], state_dict_local["model"])

    optimizer_rank0 = (
        state_dict_rank0["optimizer"]["optimizer"]
        if fp16
        else state_dict_rank0["optimizer"]
    )
    optimizer_local = (
        state_dict_local["optimizer"]["optimizer"]
        if fp16
        else state_dict_local["optimizer"]
    )

    for k, v in optimizer_local["state"].items():
        optimizer_rank0["state"][k] = v

    if fp16:
        for group_idx, param_group in enumerate(
            state_dict_local["optimizer"]["fp32_from_fp16_params"]
        ):
            for param_in_group_idx, param in enumerate(param_group):
                if param is not None:
                    state_dict_rank0["optimizer"]["fp32_from_fp16_params"][group_idx][
                        param_in_group_idx
                    ] = param

    return state_dict_rank0


def load_checkpoint(model, optimizer, lr_scheduler, load_arg="load"):
    """Load a model checkpoint and return the iteration."""

    from megatron import get_args
    from megatron import mpu
    from megatron import print_rank_last
    from megatron.checkpointing import get_checkpoint_tracker_filename
    from megatron.checkpointing import set_checkpoint_version
    from megatron.checkpointing import check_checkpoint_args
    from megatron.checkpointing import update_num_microbatches

    if mpu.get_data_parallel_rank() == 0:
        # at dp rank 0, we still follow the native load_checkpoint by megatron
        from megatron.checkpointing import load_checkpoint as load_checkpoint_native

        return load_checkpoint_native(model, optimizer, lr_scheduler, load_arg)

    args = get_args()
    load_dir = getattr(args, load_arg)

    if hasattr(model, 'module'):
        model = model.module
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        print_rank_last(
            "WARNING: could not find the metadata file {} ".format(tracker_filename)
        )
        print_rank_last(
            "    will not load any checkpoints and will start from " "random"
        )
        return 0

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration = 0
    release = False
    with open(tracker_filename, "r") as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == "release"
            if not release:
                print_rank_last(
                    "ERROR: Invalid metadata file {}. Exiting".format(tracker_filename)
                )
                sys.exit()

    assert iteration > 0 or release, "error parsing metadata file {}".format(
        tracker_filename
    )

    # Checkpoint.
    checkpoint_name_rank0 = get_fmoe_checkpoint_name(load_dir, iteration, release, 0)
    checkpoint_name_local = get_fmoe_checkpoint_name(
        load_dir, iteration, release, mpu.get_data_parallel_rank()
    )
    print_rank_last(
        " loading checkpoint at rank 0 from {} and rank {} from {} at iteration {}, will merge them later".format(
            checkpoint_name_rank0,
            mpu.get_data_parallel_rank(),
            checkpoint_name_local,
            iteration,
        )
    )

    # Load the checkpoint.
    def load_state_dict(checkpoint_name):
        try:
            state_dict = torch.load(checkpoint_name, map_location="cpu")
        except ModuleNotFoundError:
            from megatron.fp16_deprecated import loss_scaler

            # For backward compatibility.
            print_rank_last(" > deserializing using the old code structure ...")
            sys.modules["fp16.loss_scaler"] = sys.modules[
                "megatron.fp16_deprecated.loss_scaler"
            ]
            sys.modules["megatron.fp16.loss_scaler"] = sys.modules[
                "megatron.fp16_deprecated.loss_scaler"
            ]
            state_dict = torch.load(checkpoint_name, map_location="cpu")
            sys.modules.pop("fp16.loss_scaler", None)
            sys.modules.pop("megatron.fp16.loss_scaler", None)
        return state_dict

    state_dict_rank0 = load_state_dict(checkpoint_name_rank0)
    state_dict_local = load_state_dict(checkpoint_name_local)

    state_dict = merge_state_dict(state_dict_rank0, state_dict_local, args.fp16)

    # set checkpoint version
    set_checkpoint_version(state_dict.get("checkpoint_version", 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict["iteration"]
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict["total_iters"]
            except KeyError:
                print_rank_last(
                    "A metadata file exists but unable to load "
                    "iteration from checkpoint {}, exiting".format(
                        checkpoint_name_local
                    )
                )
                sys.exit()

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if "args" in state_dict:
        checkpoint_args = state_dict["args"]
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(
            checkpoint_args, "consumed_train_samples", 0
        )
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(
            checkpoint_args, "consumed_valid_samples", 0
        )
    else:
        print_rank_last("could not find arguments in the checkpoint ...")

    # Model.
    model.load_state_dict(state_dict["model"])

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            if optimizer is not None:
                optimizer.load_state_dict(state_dict["optimizer"])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        except KeyError:
            print_rank_last(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-optim or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name_local)
            )
            sys.exit()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict["random_rng_state"])
            np.random.set_state(state_dict["np_rng_state"])
            torch.set_rng_state(state_dict["torch_rng_state"])
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            mpu.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])
        except KeyError:
            print_rank_last(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name_local)
            )
            sys.exit()

    torch.distributed.barrier()
    print_rank_last(
        "  successfully loaded checkpoint (with expert parametes updated) from {} at iteration {}".format(
            args.load, iteration
        )
    )

    return iteration
