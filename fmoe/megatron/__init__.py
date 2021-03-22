r"""
A set of modules to plugin into Megatron-LM with FastMoE
"""
from .utils import add_fmoe_args

from .layers import MegatronMLP
from .layers import fmoefy

from .checkpoint import save_checkpoint
from .checkpoint import load_checkpoint

from .distributed import DistributedDataParallel
