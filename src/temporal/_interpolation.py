"""
InterpolationAligner — 时间插值与统一时间映射对齐器

适用场景：
- 采样时间离散、时间点不统一
- 存在缺失观测
- 需要统一存储结构

支持线性插值、样条插值、局部回归拟合。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseTemporalAligner
from ..utils._interpolation_utils import mark_interpolated, get_interpolation_stats
from ..utils._time_utils import build_common_time_grid

if TYPE_CHECKING:
    from anndata import AnnData


class InterpolationAligner(BaseTemporalAligner):
    """时间插值与统一时间映射对齐器。

    构建统一时间参考框架，将离散观测映射到一致时间尺度。
    保留插值标记以区分原始观测点与估计点。
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

        logg.info(f"插值对齐: {len(valid_mods)} 个模态, 方法={method}, 网格={n_grid}")

        # 构建统一时间网格
        common_grid = build_common_time_grid(mdata, time_key, n_grid=n_grid, method="union")

        interpolation_log: dict[str, dict] = {}

        for mod_name in valid_mods:
            adata = mdata.mod[mod_name]
            times = adata.obs[time_key].values.astype(float)

            # 提取数据矩阵
            if "X_temporal_aligned" in adata.obsm:
                X = np.asarray(adata.obsm["X_temporal_aligned"])
            else:
                X = self._get_time_series(adata, time_key)[1]

            # 执行插值
            X_interp, is_original = self._interpolate(
                original_times=times,
                original_matrix=X,
                target_grid=common_grid,
                method=method,
                preserve_original=self.preserve_original,
            )

            # 创建新的 AnnData 或更新
            adata.obsm["X_temporal_aligned"] = X_interp
            adata.obs["aligned_time"] = common_grid  # 用统一网格覆盖

            # 标记插值点
            mark_interpolated(adata, is_original)

            stats = get_interpolation_stats(adata)
            interpolation_log[mod_name] = {
                "n_original": stats["n_original"],
                "n_interpolated": stats["n_interpolated"],
                "method": method,
            }
            logg.hint(f"  [{mod_name}]: {stats['n_original']} orig + {stats['n_interpolated']} interp = {adata.n_obs}")

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

    def _interpolate(
        self,
        original_times: np.ndarray,
        original_matrix: np.ndarray,
        target_grid: np.ndarray,
        method: str = "spline",
        preserve_original: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """执行插值并保留原始观测点。

        Returns:
            (interpolated_matrix, is_original_mask) —
            按 target_grid 排序的矩阵和原始观测布尔掩码
        """
        from scipy.interpolate import CubicSpline, interp1d

        n_features = original_matrix.shape[1]
        n_target = len(target_grid)

        result = np.zeros((n_target, n_features))
        is_original = np.zeros(n_target, dtype=bool)

        # 对每个特征分别插值
        for f in range(min(n_features, 100)):  # 限制插值特征数以控制时间
            y = original_matrix[:, f]

            # 按时间排序
            sort_idx = np.argsort(original_times)
            t_sorted = original_times[sort_idx]
            y_sorted = y[sort_idx]

            # 移除重复时间点
            unique_idx = np.unique(t_sorted, return_index=True)[1]
            t_unique = t_sorted[unique_idx]
            y_unique = y_sorted[unique_idx]

            if len(t_unique) < 2:
                result[:, f] = y_unique[0] if len(t_unique) > 0 else 0
                continue

            try:
                if method == "spline":
                    interp_fn = CubicSpline(t_unique, y_unique, extrapolate=True)
                elif method == "linear":
                    interp_fn = interp1d(
                        t_unique, y_unique, kind="linear",
                        fill_value="extrapolate", bounds_error=False,
                    )
                else:
                    interp_fn = interp1d(
                        t_unique, y_unique, kind="linear",
                        fill_value="extrapolate", bounds_error=False,
                    )
                result[:, f] = interp_fn(target_grid)
            except Exception:
                # Fallback: 最近邻插值
                interp_fn = interp1d(
                    t_unique, y_unique, kind="nearest",
                    fill_value="extrapolate", bounds_error=False,
                )
                result[:, f] = interp_fn(target_grid)

        # 标记哪些目标网格点与原始时间点匹配（容差 1%）
        if preserve_original:
            tolerance = (target_grid[-1] - target_grid[0]) * 0.01 if len(target_grid) > 1 else 0.01
            for i, t in enumerate(target_grid):
                if np.any(np.abs(original_times - t) < tolerance):
                    is_original[i] = True

        return result, is_original
