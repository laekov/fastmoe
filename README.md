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

Step by step tutorial to install FastMoE on your local machine:

1. First of all you'll need to check your torch and nccl version, make sure to have a CUDA version compatible to the one torch was compiled (in general if you have the latest torch version it works also with the latest CUDA):
```
# go in terminal and use this command, the output should be something like this:

python -c  'import torch; print(torch.__version__); print(torch.cuda.nccl.version())'
>>> 2.0.1+cu117
>>> (2, 14, 3)  # -> this means version 2.14.3

# to check cuda version just type to see which cuda version you are using, should output something like this:

which nvcc
>>> /usr/local/cuda-11.7/bin/nvcc
```
2. An extra NCCL developer package is needed to enable the distributed features of FastMoE at the following link: https://developer.nvidia.com/nccl/nccl-legacy-downloads. Make sure to follow this steps:
```
# following the previous example I'll consider the version 2.14.3 with a CUDA version <= System CUDA version and Ubuntu 20.04
# the first command is different depending on the system and the version, just paste it from the site

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# don't forget to install the package, this command is difficult to see as it is written at the end
# outside the code block for each different installation

sudo apt install libnccl2=2.14.3-1+cuda11.7 libnccl-dev=2.14.3-1+cuda11.7 
```

3. Now you can clone the repository and enter the folder to launch the installation script as follows:
```
# clone repo and move into the folder

git clone https://github.com/laekov/fastmoe.git
cd fastmoe

# Option 1: disabling distributed features

USE_NCCL=0 python setup.py install

# Option 2: enabling distributed features

python setup.py install
```

#### Troubleshooting

If you have errors (warnings are OK) during the compilation make sure that the installer has the correct flags, this can be seen in the error as `-I/home/path/to/bin` and `-L/home/path/to/lib`. This flags should point to the correct CUDA for which all the other packages are compatible (torch and NCCL), if this paths are not correct you'll have to tell the system explicitly which CUDA version you want to use. A simple solution could be like this:
```
# add this lines at the end of your ~/.bashrc

nano ~/.bashrc
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib:$LD_LIBRARY_PATH"

# update the ~/.bashrc and try again

source ~/.bashrc
cd fastmoe && python setup.py install
```

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
model = fmoefy(model, num_experts=<number of experts per worker>)

train(model, ...)
```

A detailed tutorial to _moefy_ Megatron-LM can be found
[here](examples/megatron).

### Using FastMoE as a PyTorch module

An example MoE transformer model can be seen in the
[Transformer-XL](examples/transformer-xl) example. The easist way is to replace
the MLP layer by the `FMoE` layers.

### Using FastMoE in Parallel

FastMoE supports both data parallel and model parallel. 

#### Data Parallel

In FastMoE's data parallel mode, both the gate and the experts are replicated on each worker. 
The following figure shows the forward pass of a 3-expert MoE with 2-way data parallel.

<p align="center">
<img src="doc/fastmoe_data_parallel.png" width="600">
</p>

For data parallel, no extra coding is needed. FastMoE works seamlessly with PyTorch's `DataParallel` or `DistributedDataParallel`.
The only drawback of data parallel is that the number of experts is constrained by each worker's memory.

#### Model Parallel

In FastMoE's model parallel mode, the gate network is still replicated on each worker but
experts are placed separately across workers.
Thus, by introducing additional communication cost, FastMoE enjoys a large expert pool whose size is proportional to the number of workers.

The following figure shows the forward pass of a 6-expert MoE with 2-way model parallel. Note that experts 1-3 are located in worker 1 while experts 4-6 are located in worker 2.

<p align="center">
<img src="doc/fastmoe_model_parallel.png" width="600">
</p>

FastMoE's model parallel requires sophiscated parallel strategies that neither PyTorch nor
Megatron-LM provides. The `fmoe.DistributedGroupedDataParallel` module is
introduced to replace PyTorch's DDP module.

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
