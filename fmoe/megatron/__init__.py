r"""
A set of modules to plugin into Megatron-LM with FastMoE
"""
from .utils import add_fmoe_args

from .layers import MegatronMLP
from .layers import fmoefy

from .checkpoint import save_checkpoint
from .checkpoint import load_checkpoint

from .distributed import DistributedDataParallel

from .balance import reset_gate_hook
from .balance import get_balance_profile
from .balance import generate_megatron_gate_hook
from .balance import add_balance_log

from .patch import patch_forward_step
from .patch import patch_model_provider
