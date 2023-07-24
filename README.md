<img height='60px' src='doc/logo/rect.png'/>

[Release note](doc/release-note.md)
| [中文文档](doc/readme-cn.md)
| [Slack workspace](https://join.slack.com/t/fastmoe/shared_invite/zt-mz0ai6ol-ggov75D62YsgHfzShw8KYw)

## Introduction

An easy-to-use and efficient system to support the Mixture of Experts (MoE) 
model for PyTorch. 

## Installation

### Prerequisites

PyTorch with CUDA is required. The repository is currently tested with PyTorch
v1.10.0 and CUDA 11.3, with designed compatibility to older and newer versions.

The minimum version of supported PyTorch is `1.7.2` with CUDA `10`. However,
there are a few known issues that requires manual modification of FastMoE's
code with specific older dependents.

If the distributed expert feature is enabled, NCCL with P2P communication
support, typically versions `>=2.7.5`, is needed. 

### Installing

FastMoE contains a set of PyTorch customized opearators, including both C and
Python components. Use `python setup.py install` to easily install and enjoy
using FastMoE for training.

A step-by-step tutorial for the installation procedure can be found [here](doc/installation-guide.md).

The distributed expert feature is enabled by default. If you want to disable
it, pass environment variable `USE_NCCL=0` to the setup script.

Note that an extra NCCL developer package is needed, which has to be consistent
with your PyTorch's NCCL version, which can be inspected by running
`torch.cuda.nccl.version()`. The 
[official PyTorch docker image](https://hub.docker.com/r/pytorch/pytorch) is
recommended, as the environment is well-setup there. Otherwise, you can access
the [download link of all NCCL
versions](https://developer.nvidia.com/nccl/nccl-legacy-downloads) to download
the NCCL package that is suitable for you.

## Usage 

### FMoEfy a Transformer model

Transformer is currently one of the most popular models to be extended by MoE. Using
FastMoE, a Transformer-based model can be extended as MoE by an one-key plugin
shown as follow.

For example, when using [Megatron-LM](https://github.com/nvidia/megatron-lm),
using the following lines can help you easily scale up the MLP layers to
multiple experts.

```python
model = ...

from fmoe.megatron import fmoefy
model = fmoefy(model, fmoe_num_experts=<number of experts per worker>)

train(model, ...)
```

A detailed tutorial to _moefy_ Megatron-LM can be found
[here](examples/megatron).

### Using FastMoE as a PyTorch module

An example MoE transformer model can be seen in the
[Transformer-XL](examples/transformer-xl) example. The easist way is to replace
the MLP layer by the `FMoE` layers.

### Using FastMoE in Parallel

FastMoE supports multiple ways of parallel training. See [a comprehensive
document for parallelism](doc/parallelism) for details. Below shows the two
simplest ways of using FastMoE in parallel.

#### Data Parallel

In FastMoE's data parallel mode, both the gate and the experts are replicated on each worker. 
The following figure shows the forward pass of a 3-expert MoE with 2-way data parallel.

<p align="center">
<img src="doc/parallelism/fastmoe_data_parallel.png" width="600">
</p>

For data parallel, no extra coding is needed. FastMoE works seamlessly with PyTorch's `DataParallel` or `DistributedDataParallel`.
The only drawback of data parallel is that the number of experts is constrained by each worker's memory.

#### Expert Parallel (also called Model Parlallel in some previous versions)

In FastMoE's expert parallel mode, the gate network is still replicated on each worker but
experts are placed separately across workers.
Thus, by introducing additional communication cost, FastMoE enjoys a large expert pool whose size is proportional to the number of workers.

The following figure shows the forward pass of a 6-expert MoE with 2-way model parallel. Note that experts 1-3 are located in worker 1 while experts 4-6 are located in worker 2.

<p align="center">
<img src="doc/parallelism/fastmoe_expert_parallel.png" width="600">
</p>

FastMoE's expert parallel requires sophiscated parallel strategies that neither
PyTorch nor Megatron-LM provided when FastMoE was created. The
`fmoe.DistributedGroupedDataParallel` module is introduced to replace PyTorch's
DDP module.

#### Faster Performance Features

From a PPoPP'22 paper, _FasterMoE: modeling and optimizing training of
large-scale dynamic pre-trained models_, we have adopted techniques to make
FastMoE's model parallel much more efficient.

These optimizations are named as **Faster Performance Features**, and can be
enabled via several environment variables. Their usage and constraints are
detailed in [a separate document](doc/fastermoe).

## Citation

For the core FastMoE system.

```
@article{he2021fastmoe,
      title={FastMoE: A Fast Mixture-of-Expert Training System}, 
      author={Jiaao He and Jiezhong Qiu and Aohan Zeng and Zhilin Yang and Jidong Zhai and Jie Tang},
      journal={arXiv preprint arXiv:2103.13262},
      year={2021}
}
```

For the [faster performance features](doc/fastermoe).

```
@inproceedings{he2022fastermoe,
    author = {He, Jiaao and Zhai, Jidong and Antunes, Tiago and Wang, Haojie and Luo, Fuwen and Shi, Shangfeng and Li, Qin},
    title = {FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models},
    year = {2022},
    isbn = {9781450392044},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3503221.3508418},
    doi = {10.1145/3503221.3508418},
    booktitle = {Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
    pages = {120–134},
    numpages = {15},
    keywords = {parallelism, distributed deep learning, performance modeling},
    location = {Seoul, Republic of Korea},
    series = {PPoPP '22}
}
```

## Troubleshootings / Discussion

If you have any problem using FastMoE, or you are interested in getting involved in developing FastMoE, feel free to join [our slack channel](https://join.slack.com/t/fastmoe/shared_invite/zt-mz0ai6ol-ggov75D62YsgHfzShw8KYw).
