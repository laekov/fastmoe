Examples of FastMoE
===

As FastMoE supports both stand-alone training (or built-in data parallelism
supported by PyTorch), and expert parallelism implemented by customized
operators, we present two examples to show the usage of them separately.

### Transformer-XL

This example contains a single-process version of transformer training code
that uses PyTorch's DataParallel module to utilize multiple GPUs within one
node. In this example, FastMoE works as a simple local module without involving
any means of parallelism.

### Megatron-LM

[Megatron-LM](https://github.com/nvidia/megatron-lm) is a transformer framework
developed by NVIDIA. It supports diverse parallelisms, including data, model,
and pipeline. It is scalable to up to thousands of GPUs, with one process
binded to each GPU.

FastMoE works with any combination of the parallelisms provided by Megatron-LM.
In the example, the dimension of data parallelism is used as the communication
group for expert parallelism, so that the GPU memory consumption is kept
identical to the original non-MoE model, and the model size is enlarged.
