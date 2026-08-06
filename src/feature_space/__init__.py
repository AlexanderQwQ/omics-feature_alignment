"""特征空间校正模块（阶段二）"""

from ._base import BaseFeatureAligner
from ._mnn import MNNAligner
from ._cca import CCAAligner
from ._optimal_transport import OptimalTransportAligner
from ._manifold import ManifoldAligner
from ._selector import FeatureSpaceSelector

__all__ = [
    "BaseFeatureAligner",
    "MNNAligner",
    "CCAAligner",
    "OptimalTransportAligner",
    "ManifoldAligner",
    "FeatureSpaceSelector",
]
