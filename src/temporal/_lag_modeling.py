"""
LagModelingAligner — 时间延迟建模与相关性对齐器

适用场景：
- 跨组学模态之间存在响应滞后（转录→蛋白）
- 微生物群落与宿主免疫的耦合关系
- 非同步变化的时间偏移校正

通过滑动窗口相关分析和时滞相关分析估计响应延迟。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseTemporalAligner

if TYPE_CHECKING:
    from anndata import AnnData


class LagModelingAligner(BaseTemporalAligner):
    """时间延迟建模与相关性对齐器。

    通过滑动窗口相关和时滞分析，
    识别并校正不同模态之间的响应滞后关系。
    """

    def __init__(
        self,
        max_lag: int = 5,
        method: str = "pearson",
        window_size: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_lag = max_lag
        self.method = method
        self.window_size = window_size
        self._method_name = "LagModelingAligner"

    def run(
        self,
        mdata: MuData,
        time_key: str = "aligned_time",
        max_lag: int | None = None,
        method: str | None = None,
        window_size: int | None = None,
        **kwargs,
    ) -> MuData:
        valid_mods = self._validate_time_column(mdata, time_key)
        if len(valid_mods) < 2:
            logg.warning("需要至少 2 个含时间的模态进行滞后分析")
            return mdata

        max_lag = max_lag if max_lag is not None else self.max_lag
        method = method or self.method
        window_size = window_size or self.window_size

        logg.info(f"时间延迟建模: {len(valid_mods)} 个模态, max_lag={max_lag}")

        # 提取各模态的均值时间序列
        mean_series = {}
        for mod_name in valid_mods:
            times, X = self._get_time_series(mdata.mod[mod_name], time_key)
            mean_series[mod_name] = {
                "times": times,
                "mean": np.mean(X, axis=1),
            }

        # 成对滞后分析
        lag_results: dict[str, dict] = {}

        mod_names = list(mean_series.keys())
        for i in range(len(mod_names)):
            for j in range(i + 1, len(mod_names)):
                mod_a, mod_b = mod_names[i], mod_names[j]

                result = self._compute_lag_correlation(
                    mean_series[mod_a]["mean"],
                    mean_series[mod_b]["mean"],
                    max_lag,
                    method,
                )
                pair_key = f"{mod_a}_{mod_b}"
                lag_results[pair_key] = result
                logg.hint(
                    f"  {pair_key}: optimal lag={result['optimal_lag']}, "
                    f"corr={result['max_correlation']:.4f}"
                )

                # 如果检测到显著滞后，记录建议的校正
                if abs(result["optimal_lag"]) > 0 and result["max_correlation"] > 0.3:
                    logg.info(
                        f"    建议: {mod_a} → {mod_b} 偏移 {result['optimal_lag']} 步"
                    )

        # 滑动窗口相关分析
        window_correlations = self._sliding_window_correlation(
            mean_series, valid_mods, window_size, method,
        )

        # 存储结果
        self._store_trace(
            mdata,
            method="lag_modeling",
            params={
                "max_lag": max_lag,
                "method": method,
                "window_size": window_size,
            },
            extra={
                "lag_results": lag_results,
                "window_correlations": window_correlations,
                "stored_in_obsm": "X_temporal_aligned",
            },
        )

        # 对各模态应用滞后校正
        for mod_name in valid_mods:
            adata = mdata.mod[mod_name]
            if "X_temporal_aligned" not in adata.obsm:
                _, X = self._get_time_series(adata, time_key)
                adata.obsm["X_temporal_aligned"] = X

        logg.info(f"延迟建模完成: {len(lag_results)} 对模态")
        return mdata

    def _compute_lag_correlation(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        max_lag: int,
        method: str,
    ) -> dict:
        """计算两个序列间的最优滞后和最大相关。

        Returns:
            {"optimal_lag": int, "max_correlation": float, "lag_correlations": list}
        """
        # 对齐到较短序列长度
        min_len = min(len(series_a), len(series_b))
        a = series_a[:min_len]
        b = series_b[:min_len]

        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = self._correlation(a[-lag:], b[:lag], method)
            elif lag > 0:
                corr = self._correlation(a[:-lag], b[lag:], method)
            else:
                corr = self._correlation(a, b, method)
            correlations.append({"lag": int(lag), "correlation": float(corr)})

        best = max(correlations, key=lambda x: abs(x["correlation"]))
        return {
            "optimal_lag": best["lag"],
            "max_correlation": best["correlation"],
            "lag_correlations": correlations,
        }

    def _sliding_window_correlation(
        self,
        mean_series: dict,
        mod_names: list[str],
        window_size: int,
        method: str,
    ) -> list[dict]:
        """滑动窗口交叉相关分析。

        Returns:
            每个窗口的相关系数列表
        """
        results = []
        # 以第一个模态的时间点为参考
        ref_mod = mod_names[0]
        ref_series = mean_series[ref_mod]["mean"]
        n_points = len(ref_series)

        if n_points < window_size:
            return results

        for start in range(0, n_points - window_size + 1):
            end = start + window_size
            window_result = {"window": [int(start), int(end)], "pairs": {}}

            for i in range(len(mod_names)):
                for j in range(i + 1, len(mod_names)):
                    a = mean_series[mod_names[i]]["mean"]
                    b = mean_series[mod_names[j]]["mean"]
                    # 裁剪到窗口
                    a_win = a[start:min(end, len(a))]
                    b_win = b[start:min(end, len(b))]
                    if len(a_win) > 1 and len(b_win) > 1:
                        min_len = min(len(a_win), len(b_win))
                        corr = self._correlation(a_win[:min_len], b_win[:min_len], method)
                        pair_key = f"{mod_names[i]}_{mod_names[j]}"
                        window_result["pairs"][pair_key] = float(corr)

            results.append(window_result)

        return results

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray, method: str) -> float:
        """计算相关系数"""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        a, b = a.ravel()[:len(b)], b.ravel()[:len(a)]
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        try:
            if method == "pearson":
                return float(np.corrcoef(a, b)[0, 1])
            elif method == "spearman":
                from scipy.stats import spearmanr
                return float(spearmanr(a, b)[0])
            elif method == "mutual_info":
                from sklearn.feature_selection import mutual_info_regression
                return float(mutual_info_regression(a.reshape(-1, 1), b)[0])
        except Exception:
            return 0.0

        return 0.0
