"""多组学动态特征对齐与集成模块"""

from ._version import __version__

# Settings must be imported first
from ._settings import Verbosity, settings

# Core namespaces (scanpy/muon convention)
from . import preprocessing as da
from . import tools as tl
from . import plotting as pl

# Pipeline
from .pipeline import DynamicAlignmentPipeline

# I/O
from . import io

__all__ = [
    "__version__",
    "Verbosity",
    "settings",
    "da",
    "tl",
    "pl",
    "io",
    "DynamicAlignmentPipeline",
]
