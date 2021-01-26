Fast MoE
===

## Introduction

An easy-to-use but efficient implementation of the Mixture of Experts (MoE) 
model for PyTorch. 

## Installation

### Prerequisites

PyTorch with CUDA is required. The repository is currently tested with PyTorch
v1.6.0 and CUDA 10, with designed compatibility to other versions.

If distributed version is enabled, NCCL with P2P communication support,
typically versions >= 2.7.5 is needed. Note that the MPI backend is used as
there are some necessary messages to be passed by MPI in FMoE, the backend
should be `mpi`. However, as there are other data to be synchronized by
`torch.distributed`, cuda-aware mpi is required.

### Installing

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

## Feature Roadmap

### Better All-to-all communication efficiency and computation performance

The dispatching process from source worker to the expert is time-consuming and
topology-aware, as it is an all-to-all communication. Overlapping or other
communication reducition technologies can be applied to reduce the overhead of
this step. However, this demands much research and coding efforts.

### Dynamic expert distribution load balancing

Load imbalance is observed as there is no loss item about load balancing. Some
experts are significantly more frequently called. Therefore, a dynamic scheduler
to duplicate or recycle some experts on some workers may be effective.

### Model parallel the experts

To enable larger expert sizes. 

### Use zero-optimizer to reduce memory consumption


