Fast MoE
===

## Introduction

An easy-to-use but efficient implementation of the Mixture of Experts (MoE) 
model for PyTorch. 

## Installation

### Prerequisites

PyTorch with CUDA is required. The repository is currently tested with PyTorch
v1.8.0 and CUDA 10, with designed compatibility to older versions.

If the distributed expert feature is enabled, NCCL with P2P communication
support, typically versions `>=2.7.5`, is needed. 

### Installing

Fast MoE contains a set of PyTorch customized opearators, including both C and
Python components. Use `python setup.py install` to easily install and enjoy
using Fast MoE for training.

The distributed expert feature is disabled by default. If you want to disable
it, pass environment variable `USE_NCCL=1` to the setup script.

Note that an extra NCCL developer package is needed, which has to be consistant
with your PyTorch's NCCL version, which can be inspected by running
`torch.cuda.nccl.version()`. The [official PyTorch docker image]() is
recommended, as the environment is well-setup there. Otherwise, you can access
the [download link of all NCCL
versions](https://developer.nvidia.com/nccl/nccl-legacy-downloads) to download
the NCCL package that is suitable for you.

## Usage 

### FMoEfy a transformer model

Transformer is currently the most popular model to be extended by MoE. Using
Fast MoE, a transformer-based model can be extended as MoE by an one-key plugin
shown as follow.

For example, when using [Megatron-LM](https://github.com/nvidia/megatron-lm),
using the following lines can help you easily scale up the MLP layers to
multiple experts.

```python
model = ...

from fmoe.megatron import fmoefy
model = fmoefy(model, num_experts=<number of experts per worker>)

train(model, ...)
```

A detailed tutorial to _moefy_ Megatron-LM can be found
[here](examples/megatron).

### Using Fast MoE as a PyTorch module

An example MoE transformer model can be seen in the
[Transformer-XL](examples/transformer-xl) example. The easist way is to replace
the MLP layer by the `FMoE` layers.

### Using Fast MoE in Parallel

For data parallel, no extra coding is needed.

For expert parallel, in which experts are located separately across workers, 
which requires sophiscated data-parallel strategies that neither PyTorch nor
Megatron-LM provides. The `fmoe.DistributedGroupedDataParallel` module is
introduced to replace PyTorch's DDP module.
