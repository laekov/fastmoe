r"""
The fmoe package contains MoE Layers only.
"""

from .layers import FMoELinear, FMoENaiveGate, FMoETransformerMLP
from .distributed import DistributedGroupedDataParallel
