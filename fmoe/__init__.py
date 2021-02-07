r"""
The fmoe package contains MoE Layers only.
"""

from .layers import FMoELinear, FMoETransformerMLP
from .distributed import DistributedGroupedDataParallel
