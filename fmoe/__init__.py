r"""
The fmoe package contains MoE Layers only.
"""

from .layers import FMoELinear, FMoE
from .transformer import FMoETransformerMLP
from .distributed import DistributedGroupedDataParallel
