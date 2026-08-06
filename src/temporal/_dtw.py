"""
DTWAligner — 动态时间规整对齐器

基于 tslearn 实现 DTW/SoftDTW，支持：
- Sakoe-Chiba / Itakura 窗口约束
- 预平滑处理（降低噪声）
- 成对模态 DTW 距离矩阵
- 规整路径存储
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


class DTWAligner(BaseTemporalAligner):
    """动态时间规整（DTW）对齐器。

    在保持 n_obs 不变的前提下，计算模态间的 DTW 距离和规整路径。
    适用场景：采样密集（≥5 时间点）、局部形态相似但节奏不同的动态模式。
    """

    def __init__(
        self,
        backend: str = "tslearn",
        window_type: str = "sakoechiba",
        window_size: float = 0.1,
        metric: str = "euclidean",
        pre_smoothing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.backend = backend
        self.window_type = window_type
        self.window_size = window_size
        self.metric = metric
        self.pre_smoothing = pre_smoothing
        self._method_name = "DTWAligner"

    def run(
        self,
        mdata: MuData,
        time_key: str = "aligned_time",
        window_type: str | None = None,
        window_size: float | None = None,
        metric: str | None = None,
        **kwargs,
    ) -> MuData:
        valid_mods = self._validate_time_column(mdata, time_key)
        if len(valid_mods) < 1:
            logg.warning("没有模态包含时间信息，跳过 DTW 对齐")
            return mdata

        window_type = window_type or self.window_type
        window_size = window_size if window_size is not None else self.window_size
        metric = metric or self.metric

        logg.info(f"DTW 对齐: {len(valid_mods)} 个模态, 窗口={window_type}({window_size})")

        # 提取每个模态的时间序列（按时间聚合的均值向量）
        series = {}
        for mod_name in valid_mods:
            times, X = self._get_time_series(mdata.mod[mod_name], time_key)
            if self.pre_smoothing:
                X = self._smooth(X)
            series[mod_name] = {"times": times, "matrix": X}

        # 计算成对 DTW 距离和规整路径
        dtw_distances: dict[str, float] = {}
        warping_paths: dict[str, list] = {}

        mod_names = list(series.keys())
        for i in range(len(mod_names)):
            for j in range(i + 1, len(mod_names)):
                mod_a, mod_b = mod_names[i], mod_names[j]
                dist, path = self._pairwise_dtw(
                    series[mod_a]["matrix"],
                    series[mod_b]["matrix"],
                    window_type,
                    window_size,
                    metric,
                )
                pair_key = f"{mod_a}_{mod_b}"
                dtw_distances[pair_key] = float(dist)
                warping_paths[pair_key] = path
                logg.hint(f"  {pair_key}: DTW distance = {dist:.4f}")

        # 将原始 X_corrected 存入 X_temporal_aligned（保持 n_obs 不变）
        for mod_name in valid_mods:
            adata = mdata.mod[mod_name]
            if "X_corrected" in adata.obsm:
                adata.obsm["X_temporal_aligned"] = np.asarray(adata.obsm["X_corrected"])
            elif "X_temporal_aligned" not in adata.obsm:
                X = self._get_time_series(adata, time_key)[1]
                adata.obsm["X_temporal_aligned"] = X

        # 存储对齐结果
        self._store_trace(
            mdata,
            method="dtw",
            params={
                "backend": self.backend,
                "window_type": window_type,
                "window_size": window_size,
                "metric": metric,
                "pre_smoothing": self.pre_smoothing,
            },
            extra={
                "n_modalities": len(valid_mods),
                "dtw_distance_matrix": dtw_distances,
                "warping_paths": warping_paths,
                "stored_in_obsm": "X_temporal_aligned",
            },
        )

        logg.info(f"DTW 对齐完成: {len(dtw_distances)} 对模态")
        return mdata

    def _pairwise_dtw(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray,
        window_type: str,
        window_size: float,
        metric: str,
    ) -> tuple[float, list]:
        """计算两个矩阵之间的 DTW 距离"""
        try:
            from tslearn.metrics import dtw_path as ts_dtw_path

            n_features = min(X_a.shape[1], X_b.shape[1])
            max_len = max(X_a.shape[0], X_b.shape[0])
            global_constraint = None
            if window_type == "sakoechiba":
                global_constraint = max(1, int(window_size * max_len))
            elif window_type == "itakura":
                global_constraint = "itakura"

            total_dist = 0.0
            best_path = []
            n_feat_to_use = min(n_features, 10)
            for f in range(n_feat_to_use):
                try:
                    path, dist = ts_dtw_path(
                        X_a[:, f].reshape(-1, 1),
                        X_b[:, f].reshape(-1, 1),
                        global_constraint=global_constraint,
                    )
                    total_dist += dist
                    if not best_path:
                        best_path = [(int(p[0]), int(p[1])) for p in path]
                except Exception:
                    pass

            avg_dist = total_dist / n_feat_to_use if n_feat_to_use > 0 else 0.0
            return avg_dist, best_path

        except ImportError:
            logg.warning("tslearn 未安装，使用 scipy fallback")
            return self._fallback_dtw(X_a, X_b)

    def _fallback_dtw(self, X_a: np.ndarray, X_b: np.ndarray) -> tuple[float, list]:
        """使用 scipy 的简化 DTW fallback（1D 序列）"""
        from scipy.spatial.distance import cdist

        seq_a = np.mean(X_a, axis=1) if X_a.ndim > 1 else X_a
        seq_b = np.mean(X_b, axis=1) if X_b.ndim > 1 else X_b

        n, m = len(seq_a), len(seq_b)
        cost = cdist(seq_a.reshape(-1, 1), seq_b.reshape(-1, 1), metric="euclidean")

        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                diag, up, left = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
                if diag <= up and diag <= left:
                    i -= 1; j -= 1
                elif up <= left:
                    i -= 1
                else:
                    j -= 1

        return float(D[n, m]), path[::-1]

    @staticmethod
    def _smooth(X: np.ndarray, window: int = 3) -> np.ndarray:
        """简单移动平均平滑"""
        if X.shape[0] < window:
            return X
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(X.astype(float), size=window, axis=0)
