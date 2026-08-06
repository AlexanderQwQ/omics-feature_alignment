"""
BaseTemporalAligner — 时间/过程对齐抽象基类

所有时间对齐方法遵循 run(mdata, time_key, **kwargs) -> MuData 契约。
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


class BaseTemporalAligner(ABC):
    """时间/过程对齐抽象基类。

    子类必须实现 run() 方法。
    所有实现应在 MuData 上原地操作并返回，同时写入处理溯源到 .uns["alignment"]["temporal"]。
    """

    def __init__(self, **kwargs) -> None:
        self._params = kwargs
        self._method_name: str = self.__class__.__name__

    @abstractmethod
    def run(self, mdata: MuData, time_key: str = "time", **kwargs) -> MuData:
        """执行时间/过程对齐。

        Args:
            mdata: 输入 MuData（含各模态 AnnData）
            time_key: .obs 中的时间列名
            **kwargs: 覆盖构造参数

        Returns:
            对齐后的 MuData（原地修改）
        """
        ...

    def _validate_time_column(self, mdata: MuData, time_key: str) -> list[str]:
        """校验哪些模态有时间列。

        Returns:
            有时间列的模态名列表
        """
        valid = []
        for mod_name, adata in mdata.mod.items():
            if time_key in adata.obs.columns:
                valid.append(mod_name)
            else:
                logg.warning(f"[{mod_name}] 缺少 '{time_key}' 列，跳过时间对齐")
        return valid

    def _get_time_series(
        self, adata: AnnData, time_key: str, layer: str = "X_corrected"
    ) -> tuple[np.ndarray, np.ndarray]:
        """从 AnnData 中提取按时间排序的聚合表达矩阵。

        Returns:
            (sorted_times, sorted_matrix) — 按时间排序的数组
        """
        times = adata.obs[time_key].values.astype(float)

        # 提取数据矩阵
        if layer in adata.obsm:
            X = np.asarray(adata.obsm[layer])
        elif layer in adata.layers:
            X = np.asarray(adata.layers[layer].toarray()
                          if hasattr(adata.layers[layer], 'toarray')
                          else adata.layers[layer])
        else:
            X = np.asarray(adata.X.toarray()
                          if hasattr(adata.X, 'toarray')
                          else adata.X)

        # 按时间排序
        sorted_idx = np.argsort(times)
        return times[sorted_idx], X[sorted_idx]

    def _store_trace(
        self,
        mdata: MuData,
        method: str,
        params: dict,
        extra: dict | None = None,
    ) -> None:
        """写入时间对齐溯源信息到 mdata.uns"""
        if "alignment" not in mdata.uns:
            mdata.uns["alignment"] = {}

        mdata.uns["alignment"]["temporal"] = {
            "method": method,
            "timestamp": str(datetime.now(timezone.utc)),
            "parameters": params,
            **(extra or {}),
        }

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self._method_name}({params_str})"
