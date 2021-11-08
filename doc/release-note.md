## v0.3.0

### FMoE core

* Previous `mp_group` is renamed to `slice_group`, indicating that all workers in the group receive the same input batch, and process a slice of the input. `mp_group` will be deprecated in our next release.
* ROCm supported.
* `FMoELinear` is moved to a stand-alone file.

### Groupped data parallel

* Support any group name by their relative tag name.

###  Load balancing

* A brand new balancing strategy - SWIPE. Contributed by authors of a (currently unpublished) paper.
* A property `has_loss` is added to each gate, in order to identify whether balance loss should be collected.

### Megatron-LM support

* Experts are partitioned by tensor model parallelism in `mp_group`, instead of expert parallelism.
* Support arbitrary customized gate in `MegatronMLP`.
* Move the patches to a stand-alone file.

### Tests

* Move util functions into `test_ddp.py`.

## v0.2.1

## Load balancing

* Fix gradient for balance loss.

### Misc

* Typos.
* Update benchmark interface.
* Remove some redundant code for performance improvement.
* Enable `USE_NCCL` by default.
* Compatibility for PyTorch `<1.8.0` and `>=1.8.0`.

### Megatron adaption

* Patch for numerical correctness of gradient clipping.
* Support to pipeline parallelism.

## v0.2.0

## Load balancing

* A brand new gate module with capacity-related utilities.
* GShard's and Switch Transformer's balance strategies are implemented as integrated gates.
* Balance loss is enabled.
* Balance monitor is provided.

## Checkpointing

* MoE models can be loaded and saved by fmoe's checkpointing module.

## Performance

* FP16 training performance is improved.

## Misc

* CUDA code directory is reconstructed.
* More tests are added.

## v0.1.2

### Compilation

- Remove dependency on the CUDA examples repository.

### Distributed

- Fix a bug related to PyTorch v1.8.0. FastMoE can now operate on multiple GPUs
on multiple nodes with PyTorch v1.8.0.

### Misc

- Fix tons of typos.
- Format the code.

## v0.1.1

### Distributed

- Broadcast data-parallel parameters before training.

### Megatron adaption

- Initialize `FMoELinear` parameters using different seed in model parallel even using the same random seed in megatron.
- Use proper comm for mp and dp.

### Transformer-XL example

- Improve scripts.

### Misc

- Logo and slack workspace link.
- Document in Chinese.
- Figures to explain how FastMoE works.

## v0.1.0

### Functions

- A model-injection-style easy-to-use user interface for Megatron-LM. 
- Support both data parallel and model parallel, and a hybrid of the two,
- Provide a new customized DDP module to synchronize in different comm groups.
- Support to customized `nn.Module` as an expert.

### Document and infrastructure

- Use PyTest.
- Setup PyLint.
- Installation and usage guide.
- Explanation of functions and code structure in code.

### Performance

- A benchmark to compare FastMoE and old PyTorch impl.
