FastMoE works with different versions of
[Megatron-LM](https://github.com/nvidia/megatron-lm).
See `fmoe/megatron/utils.py` for arguments of FastMoE.

An example patch is provided for `v2.2` release.
The patch can be directly applied to add FastMoE support if you are using
Megatron-LM v2.2.
Otherwise, you may need to manually enable FastMoE in your codebase.
The patch includes the following modifications.

### Add arguments to Megatron's argparser

In `megatron/arguments.py`, add `_add_fmoe_args` to the parser.

### Patch checkpoint

In `megatron/training.py`, replace `load_checkpoint` and `save_checkpoint` by
functions with the same name in `fmoe.megatron.checkpointing`.

### Building the model in FastMoE style

In `megatron/training.py`, the `fmoe.megatron.fmoefy` function is used as an
entrance to one-key introduce FastMoE layer to replace the MLP layers in the
transformer language models.

```python
from fmoe.megatron import fmoefy
model = fmoefy(model, num_experts=4)
```

Note that the `fmoefy` function currently only takes a standard Megatron-LM's
top-level raw model as input, i.e. the MLP layers should be available at
`model.language_model.transformer.layers[i].mlp`.

### Using FastMoE's model parallellization

In `megatron/training.py`, the `LocalDDP` module is replaced by the one in 
`fmoe.megatron` to enable the sophiscated data parallel strategies that can
parallelize the experts across both the data parallel group and the (tensor) 
model parallel model group.

```python
# from megatron.model import DistributedDataParallel as LocalDDP
from fmoe.megatron import DistributedDataParallel as LocalDDP
```

### Fix gradient clipping

Megatron-LM uses gradient normalization, which is incompatible with FastMoE.
Incorrect norm of the gradients lead to inconsistent parameter updates.
Apply `clip-grad-v2.2.patch` to fix the issue.

Note that only 2-norm is implemented in the patch. If other norm methods is
used, remember to implement it accordingly.

### Train as usual

Start traning with FastMoE by using the scripts provided by Megatron-LM.
