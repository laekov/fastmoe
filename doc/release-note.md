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
