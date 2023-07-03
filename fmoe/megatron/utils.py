r"""
Utility in Megatron
"""

import argparse

def add_fmoe_args(parser):
    group = parser.add_argument_group(title="fastmoe")

    group.add_argument("--fmoefy", action="store_true")
    try:
        group.add_argument("--num-experts", type=int, default=None)
    except argparse.ArgumentError:
        group.add_argument("--fmoe-num-experts", type=int, default=None)
    group.add_argument("--top-k", type=int, default=2)
    group.add_argument("--balance-loss-weight", type=float, default=1)
    group.add_argument("--balance-strategy", type=str, default=None)
    group.add_argument("--hidden-hidden-size", type=int, default=None)

    return parser
