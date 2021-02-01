A modified version of Megatron-LM that can cope with FastMoE can be found in 
[this repository](https://github.com/laekov/fmoe-megatron).

Using `fmoe.megatron.create_moe_mlp` to replace the `ParallelMLP` module in 
Megatron's transformer model is all you need. 

In our fork, the required modifications are located at line 425 of
`megatron/model/transformer.py` as follow.

```Python
        # MLP
        if args.num_experts == 1:
            self.mlp = ParallelMLP(init_method,
                    output_layer_init_method)
        else:
            from fmoe.megatron import create_moe_mlp
            self.mlp = create_moe_mlp(args)

```

When properly added `--num-experts` argument to `megatron/arguments.py`, FastMoE
is enabled without extra burden.
