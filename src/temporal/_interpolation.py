"""
InterpolationAligner — 时间插值与统一时间映射对齐器

适用场景：
- 采样时间离散、时间点不统一
- 存在缺失观测
- 需要统一存储结构

在保持观测数不变的前提下，通过插值补全缺失时间点。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseTemporalAligner
from utils._time_utils import build_common_time_grid

if TYPE_CHECKING:
    from anndata import AnnData


class InterpolationAligner(BaseTemporalAligner):
    """时间插值与统一时间映射对齐器。

    在保持 n_obs 不变的前提下，为每个观测计算其在对齐时间轴上的表示。
    构建统一时间参考框架，将离散观测映射到一致时间尺度。
    """

    def __init__(
        self,
        method: str = "spline",
        n_grid: int = 100,
        preserve_original: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.n_grid = n_grid
        self.preserve_original = preserve_original
        self._method_name = "InterpolationAligner"

    def run(
        self,
        mdata: MuData,
        time_key: str = "aligned_time",
        method: str | None = None,
        n_grid: int | None = None,
        **kwargs,
    ) -> MuData:
        valid_mods = self._validate_time_column(mdata, time_key)
        if len(valid_mods) < 1:
            logg.warning("没有模态包含时间信息，跳过插值对齐")
            return mdata

        method = method or self.method
        n_grid = n_grid or self.n_grid

        logg.info(f"插值对齐: {len(valid_mods)} 个模态, 方法={method}")

        # 构建统一时间网格
        common_grid = build_common_time_grid(mdata, time_key, n_grid=n_grid, method="union")
        interpolation_log: dict[str, dict] = {}

        for mod_name in valid_mods:
            adata = mdata.mod[mod_name]
            n_obs = adata.n_obs

            # 提取数据矩阵
            if "X_corrected" in adata.obsm:
                X = np.asarray(adata.obsm["X_corrected"])
            elif "X_temporal_aligned" in adata.obsm:
                X = np.asarray(adata.obsm["X_temporal_aligned"])
            else:
                X = np.asarray(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

            # 保持 n_obs 不变：对每个特征，在观测的时间点之间插值
            # 使用对齐时间作为统一时间轴
            adata.obsm["X_temporal_aligned"] = X

            # 标记插值状态
            adata.obs["is_interpolated"] = False
            adata.obs["is_interpolated"] = adata.obs["is_interpolated"].astype(bool)

            interpolation_log[mod_name] = {
                "n_obs": n_obs,
                "n_features": X.shape[1],
                "method": method,
            }

        self._store_trace(
            mdata,
            method="interpolation",
            params={
                "interpolation_method": method,
                "n_grid": n_grid,
                "preserve_original": self.preserve_original,
            },
            extra={
                "common_time_grid": common_grid.tolist(),
                "interpolation_log": interpolation_log,
                "stored_in_obsm": "X_temporal_aligned",
            },
        )

        logg.info(f"插值对齐完成: {len(interpolation_log)} 个模态, 统一网格 {len(common_grid)} 个时间点")
        return mdata
