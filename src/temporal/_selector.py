"""
TemporalSelector — 时间对齐方法自动选择器

根据数据特征自动选择最优的时间/过程对齐方法：
- DTW: 采样点 ≥5, 时间连续, 局部错位
- Interpolation: 采样点 <5 或时间点不统一
- Pseudotime: 单细胞数据, 无统一采样时间
- Lag: 跨模态存在响应滞后
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseTemporalAligner
from ._dtw import DTWAligner
from ._interpolation import InterpolationAligner
from ._pseudotime import PseudotimeAligner
from ._lag_modeling import LagModelingAligner

if TYPE_CHECKING:
    pass


class TemporalSelector:
    """自动选择并执行时间/过程对齐方法。

    选择启发式：
    - 采样点 ≥5, 时间连续 → DTW
    - 采样点 <5 或时间点不统一 → Interpolation
    - 单细胞数据 → Pseudotime
    - 跨模态响应滞后 → Lag Modeling
    - 多模态混合 → 自动组合 DTW+Interpolation 或 Pseudotime+Lag
    """

    def __init__(self) -> None:
        self._available = {
            "dtw": DTWAligner,
            "interpolation": InterpolationAligner,
            "pseudotime": PseudotimeAligner,
            "lag": LagModelingAligner,
        }

    def select(self, mdata: MuData, time_key: str = "time") -> str:
        """检查数据特征，返回推荐的方法名。

        Returns:
            "dtw" | "interpolation" | "pseudotime" | "lag"
        """
        n_time_modalities = 0
        total_time_points = 0
        time_point_counts: list[int] = []
        has_large_n_obs = False

        for mod_name, adata in mdata.mod.items():
            if time_key in adata.obs.columns:
                n_time_modalities += 1
                n_unique = len(np.unique(adata.obs[time_key].values))
                total_time_points += n_unique
                time_point_counts.append(n_unique)

            # P1-9: 更准确的单细胞检测（大观测数 + 无时间 + 无批次）
            if (adata.n_obs > 500
                    and time_key not in adata.obs.columns
                    and "batch" not in adata.obs.columns):
                has_large_n_obs = True

        # P1-9: 检测跨模态响应滞后
        has_lag = self._detect_lag(mdata, time_key)

        # 决策逻辑
        if has_large_n_obs and n_time_modalities == 0:
            logg.info("TemporalSelector: 选择 pseudotime（大观测数, 无统一采样时间）")
            return "pseudotime"

        if n_time_modalities >= 2:
            avg_points = total_time_points / n_time_modalities
            # 检查时间分布是否均衡：各模态时间点数的 max/min
            if time_point_counts:
                min_pts, max_pts = min(time_point_counts), max(time_point_counts)
                balanced = min_pts > 0 and (max_pts / min_pts) <= 3.0
            else:
                balanced = True

            if avg_points >= 5 and balanced:
                if has_lag:
                    logg.info(f"TemporalSelector: 选择 dtw+lag（avg={avg_points:.1f} pts, 检测到跨模态滞后）")
                    return "lag"
                logg.info(f"TemporalSelector: 选择 dtw（平均 {avg_points:.1f} 个时间点, 分布均衡）")
                return "dtw"
            else:
                logg.info(f"TemporalSelector: 选择 interpolation（平均 {avg_points:.1f} 个时间点）")
                return "interpolation"

        if n_time_modalities >= 1:
            logg.info("TemporalSelector: 选择 interpolation")
            return "interpolation"

        logg.info("TemporalSelector: 回退到 pseudotime")
        return "pseudotime"

    def _detect_lag(self, mdata: MuData, time_key: str) -> bool:
        """检测跨模态是否存在显著的响应滞后。

        对每对模态的均值时间序列计算滞后相关，
        若最优滞后 > 1 且相关系数 > 0.3，认为存在滞后。
        """
        import numpy as np

        valid_mods = [
            name for name, adata in mdata.mod.items()
            if time_key in adata.obs.columns
        ]
        if len(valid_mods) < 2:
            return False

        # 提取各模态按时间排序的均值序列
        series = {}
        for mod_name in valid_mods:
            adata = mdata.mod[mod_name]
            times = adata.obs[time_key].values.astype(float)
            sorted_idx = np.argsort(times)
            if "X_corrected" in adata.obsm:
                X = np.asarray(adata.obsm["X_corrected"])[sorted_idx]
            elif "X_temporal_aligned" in adata.obsm:
                X = np.asarray(adata.obsm["X_temporal_aligned"])[sorted_idx]
            else:
                X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                X = np.asarray(X_data)[sorted_idx]
            series[mod_name] = np.mean(X, axis=1)

        mod_names = list(series.keys())
        for i in range(len(mod_names)):
            for j in range(i + 1, len(mod_names)):
                a = series[mod_names[i]]
                b = series[mod_names[j]]
                min_len = min(len(a), len(b))
                a, b = a[:min_len], b[:min_len]
                if min_len < 5:
                    continue

                # 简单滞后检测
                best_lag, best_corr = 0, 0.0
                for lag in range(1, min(4, min_len // 2)):
                    try:
                        corr = float(np.corrcoef(a[lag:], b[:-lag])[0, 1])
                        if abs(corr) > abs(best_corr):
                            best_corr = abs(corr)
                            best_lag = lag
                    except Exception:
                        pass

                if best_lag > 1 and best_corr > 0.3:
                    logg.hint(
                        f"  Lag detection: {mod_names[i]}_{mod_names[j]} "
                        f"lag={best_lag}, corr={best_corr:.3f}"
                    )
                    return True

        return False

    def run(
        self,
        mdata: MuData,
        method: str | None = None,
        time_key: str = "time",
        **kwargs,
    ) -> MuData:
        """自动选择并执行时间对齐。

        Args:
            mdata: 输入 MuData
            method: 显式指定方法名（None=自动选择）
            time_key: 时间列名
            **kwargs: 传递给具体对齐器的参数

        Returns:
            对齐后的 MuData
        """
        # 若指定了 lag 方法，同时执行时间对齐 + 滞后分析
        if method == "lag":
            method_name = self.select(mdata, time_key)
            if method_name != "lag":
                # 先执行基础时间对齐
                primary = self._available.get(method_name, InterpolationAligner)()
                mdata = primary.run(mdata, time_key=time_key, **kwargs)

            # 再执行滞后分析
            lag_aligner = LagModelingAligner(**kwargs.get("lag_modeling", {}))
            mdata = lag_aligner.run(mdata, time_key="aligned_time", **kwargs)
            return mdata

        if method is None:
            method = self.select(mdata, time_key)

        if method not in self._available:
            logg.warning(f"不支持的方法 '{method}'，回退到 interpolation")
            method = "interpolation"

        aligner_cls = self._available[method]
        aligner = aligner_cls(**kwargs.get(method, {}))
        return aligner.run(mdata, time_key=time_key, **kwargs)
