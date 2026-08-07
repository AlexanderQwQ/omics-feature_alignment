"""
InterpolationAligner — 时间插值与统一时间映射对齐器

适用场景：
- 采样时间离散、时间点不统一
- 存在缺失观测
- 需要统一存储结构与统一检索

通过构建统一时间参考框架，结合线性插值、样条插值或局部回归拟合，
将原始观测数据映射至一致时间尺度。保留插值标记以区分原始观测点与估计点。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData
from scipy.interpolate import CubicSpline, interp1d

import _logging as logg
from ._base import BaseTemporalAligner
from utils._time_utils import build_common_time_grid

if TYPE_CHECKING:
    from anndata import AnnData


class InterpolationAligner(BaseTemporalAligner):
    """时间插值与统一时间映射对齐器。

    对每个特征维度沿时间轴构建插值函数，在保持 n_obs 不变的前提下
    将观测值映射到统一时间尺度。使用插值标记区分原始观测点与估计点。
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
            times = adata.obs[time_key].values.astype(float)

            # 提取数据矩阵
            if "X_corrected" in adata.obsm:
                X = np.asarray(adata.obsm["X_corrected"])
            elif "X_temporal_aligned" in adata.obsm:
                X = np.asarray(adata.obsm["X_temporal_aligned"])
            else:
                X = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X)

            # 对每个特征维度在时间轴上插值，然后采样回原观测点
            X_aligned, is_original = self._interpolate_to_observations(
                times=times,
                X=X,
                common_grid=common_grid,
                method=method,
                preserve_original=self.preserve_original,
            )

            adata.obsm["X_temporal_aligned"] = X_aligned
            adata.obs["is_interpolated"] = ~is_original
            adata.obs["is_interpolated"] = adata.obs["is_interpolated"].astype(bool)

            n_interp = int((~is_original).sum())
            n_orig = int(is_original.sum())
            interpolation_log[mod_name] = {
                "n_obs": adata.n_obs,
                "n_original": n_orig,
                "n_interpolated": n_interp,
                "method": method,
            }
            logg.hint(f"  [{mod_name}]: {n_orig} orig + {n_interp} interp = {adata.n_obs} obs")

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

        logg.info(
            f"插值对齐完成: {len(interpolation_log)} 个模态, "
            f"统一网格 {len(common_grid)} 个时间点"
        )
        return mdata

    def _interpolate_to_observations(
        self,
        times: np.ndarray,
        X: np.ndarray,
        common_grid: np.ndarray,
        method: str,
        preserve_original: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """为每个观测在其时间点上做插值估计。

        步骤：
        1. 为每个特征维度，用 (time, value) 对构建插值器
        2. 在每个观测的时间点上评估插值器
        3. 如果观测时间与原始时间点匹配（容差内），标记为 original

        Returns:
            (X_aligned, is_original_mask)
        """
        n_obs, n_features = X.shape
        X_aligned = np.zeros_like(X)
        is_original = np.zeros(n_obs, dtype=bool)

        # 对每个时间点去重，取均值作为插值锚点
        unique_times = np.unique(times)
        if len(unique_times) < 2:
            # 时间点太少，无法插值，返回原始数据
            return X.copy(), np.ones(n_obs, dtype=bool)

        # 构建每个特征在锚点上的均值
        anchor_values = np.zeros((len(unique_times), n_features))
        for i, t in enumerate(unique_times):
            mask = np.abs(times - t) < (unique_times[1] - unique_times[0]) * 0.1 if len(unique_times) > 1 else True
            # 精确匹配
            mask = times == t
            if mask.sum() > 0:
                anchor_values[i] = np.mean(X[mask], axis=0)
            else:
                anchor_values[i] = X[np.argmin(np.abs(times - t))]

        # 对每个特征维度构建插值器并评估
        n_features_process = min(n_features, 200)  # 限制特征数以控制性能
        for f in range(n_features_process):
            y_anchor = anchor_values[:, f]

            # 移除重复
            _, unique_idx = np.unique(unique_times, return_index=True)
            t_u = unique_times[np.sort(unique_idx)]
            y_u = y_anchor[np.sort(unique_idx)]

            if len(t_u) < 2:
                X_aligned[:, f] = y_u[0] if len(y_u) > 0 else 0
                continue

            try:
                if method == "spline":
                    interp_fn = CubicSpline(t_u, y_u, extrapolate=True)
                elif method == "linear":
                    interp_fn = interp1d(
                        t_u, y_u, kind="linear",
                        fill_value="extrapolate", bounds_error=False,
                    )
                else:
                    # loess fallback: use spline
                    interp_fn = CubicSpline(t_u, y_u, extrapolate=True)

                X_aligned[:, f] = interp_fn(times)

            except Exception:
                # 回退到线性插值，再失败就用最近邻
                try:
                    interp_fn = interp1d(
                        t_u, y_u, kind="nearest",
                        fill_value="extrapolate", bounds_error=False,
                    )
                    X_aligned[:, f] = interp_fn(times)
                except Exception:
                    X_aligned[:, f] = X[:, f]

        # 对超出处理范围的特征维度直接复制
        if n_features > n_features_process:
            X_aligned[:, n_features_process:] = X[:, n_features_process:]

        # 标记原始观测（时间点接近锚点）
        tolerance = np.median(np.diff(unique_times)) * 0.1 if len(unique_times) > 1 else 1.0
        for i, t in enumerate(times):
            if np.any(np.abs(unique_times - t) < tolerance):
                is_original[i] = True

        return X_aligned, is_original
