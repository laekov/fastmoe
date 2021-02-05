Fast MoE currently works with the `v2.0` release of
[Megatron-LM](https://github.com/nvidia/megatron-lm).

A [patch](moefy.patch) is used to easily enable MoE in Megatron-LM for training
Bert.

The patch works in the following way.

### Building the model

In `pretrain_bert.py`, the `fmoe.megatron.fmoefy` function is used as an
entrance to one-key introduce Fast MoE layer to replace the MLP layers in the
transformer language models.

```python
from fmoe.megatron import fmoefy
model = fmoefy(model, num_experts=4)
```

Note that the `fmoefy` function currently only takes a standard Megatron-LM's
top-level raw model as input, i.e. the MLP layers should be available at
`model.language_model.transformer.layers[i].mlp`.

### Using expert parallellization

In `megatron/training.py`, the `LocalDDP` module is replaced by the one in 
`fmoe.megatron` to enable the sophiscated data parallel strategies that can
parallelize the experts across both the data parallel group and the (tensor) 
model parallel model group.

```python
# from megatron.model import DistributedDataParallel as LocalDDP
from fmoe.megatron import DistributedDataParallel as LocalDDP
```

### Train as usual

Start traning with Fast MoE by using the scripts provided by Megatron-LM.
