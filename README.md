Fast MoE
===

## Introduction

An easy-to-use but efficient implementation of the Mixture of Experts (MoE) 
model for PyTorch. 

## Installation

PyTorch with CUDA is supported. The repository is currently tested with PyTorch
v1.6.0 and CUDA 10, with designed compatibility to other versions.

Fast MoE contains a set of PyTorch customized opearators, including both C and
Python components. Use `python setup.py install` to easily install and enjoy
using Fast MoE for training.

## Usage 

### Using Fast MoE as a PyTorch module

Examples can be seen in [examples](examples/). The easist way is to replace the
feed forward layer by the `FMoE` layer.

### Using Fast MoE in Parallel

For data parallel, nothing else is needed.

For expert parallel, in which experts are located separately across workers,
NCCL and MPI backend are required to be built with PyTorch. Use environment
variable `USE_NCCL=1` to `setup.py` to enable distributing experts across
workers. Note that the arguments of the MoE layers should then be excluded from
the data parallel parameter synchronization list.
