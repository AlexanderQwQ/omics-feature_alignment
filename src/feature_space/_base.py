"""
BaseFeatureAligner — 特征空间校正抽象基类

所有特征空间方法遵循 run(mdata, **kwargs) -> MuData 契约。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class BaseFeatureAligner(ABC):
    """特征空间校正抽象基类。

    子类必须实现 run() 方法。
    在时间对齐的基础上，校正批次/模态差异引起的分布偏移。
    """

    def __init__(self, **kwargs) -> None:
        self._params = kwargs
        self._method_name: str = self.__class__.__name__

    @abstractmethod
    def run(self, mdata: MuData, **kwargs) -> MuData:
        """执行特征空间校正。

        Args:
            mdata: 输入 MuData（通常已完成时间对齐）
            **kwargs: 覆盖构造参数

        Returns:
            校正后的 MuData（原地修改）
        """
        ...

    def _get_feature_matrix(
        self, adata: AnnData, prefer_key: str = "X_temporal_aligned"
    ) -> np.ndarray:
        """从 AnnData 提取特征矩阵，优先级：
        X_temporal_aligned > X_corrected > normalized > .X
        """
        if prefer_key in adata.obsm:
            return np.asarray(adata.obsm[prefer_key])
        if "X_corrected" in adata.obsm:
            return np.asarray(adata.obsm["X_corrected"])
        if "normalized" in adata.layers:
            m = adata.layers["normalized"]
            return m.toarray() if hasattr(m, 'toarray') else np.asarray(m)
        m = adata.X
        return m.toarray() if hasattr(m, 'toarray') else np.asarray(m)

    def _store_trace(
        self,
        mdata: MuData,
        method: str,
        params: dict,
        extra: dict | None = None,
    ) -> None:
        """写入特征空间校正溯源信息"""
        if "alignment" not in mdata.uns:
            mdata.uns["alignment"] = {}

        mdata.uns["alignment"]["feature_space"] = {
            "method": method,
            "timestamp": str(datetime.now(timezone.utc)),
            "parameters": params,
            **(extra or {}),
        }

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self._method_name}({params_str})"
