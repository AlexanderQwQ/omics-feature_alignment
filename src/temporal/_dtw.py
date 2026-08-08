"""
DTWAligner — 动态时间规整对齐器

基于 tslearn 实现 DTW/SoftDTW，支持：
- Sakoe-Chiba / Itakura 窗口约束
- 加权 DTW（按时间点重要性加权）
- 预平滑处理
- 规整路径存储与实际时间轴变换
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseTemporalAligner

if TYPE_CHECKING:
    from anndata import AnnData


class DTWAligner(BaseTemporalAligner):
    """动态时间规整（DTW）对齐器。

    适用场景：采样密集（≥5 时间点）、局部形态相似但节奏不同的动态模式。
    支持约束窗口和加权 DTW 以提升对齐稳定性。
    """

    def __init__(
        self,
        window_type: str = "sakoechiba",
        window_size: float = 0.1,
        metric: str = "euclidean",
        pre_smoothing: bool = False,
        smooth_window: int = 3,
        weights: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.window_type = window_type
        self.window_size = window_size
        self.metric = metric
        self.pre_smoothing = pre_smoothing
        self.smooth_window = smooth_window
        self.weights = weights
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

        # 计算成对 DTW
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

        # 选择一个参考模态（时间点最多的），用 DTW 规整其他模态
        ref_mod = max(valid_mods, key=lambda m: len(np.unique(series[m]["times"])))

        for mod_name in valid_mods:
            adata = mdata.mod[mod_name]
            X = series[mod_name]["matrix"]

            if mod_name == ref_mod:
                adata.obsm["X_temporal_aligned"] = X
                continue

            # 用 DTW 路径将数据映射到自身时间轴（保持 n_obs 不变）
            # 对非参考模态：通过规整路径重新排序数据，使其与参考时间线对齐
            pair_key = f"{ref_mod}_{mod_name}"
            if pair_key not in warping_paths:
                pair_key = f"{mod_name}_{ref_mod}"

            if pair_key in warping_paths and warping_paths[pair_key]:
                X_aligned = self._reorder_by_warping(X, warping_paths[pair_key])
                adata.obsm["X_temporal_aligned"] = X_aligned
            else:
                adata.obsm["X_temporal_aligned"] = X

        self._store_trace(
            mdata,
            method="dtw",
            params={
                "window_type": window_type,
                "window_size": window_size,
                "metric": metric,
                "pre_smoothing": self.pre_smoothing,
                "weighted": self.weights is not None,
            },
            extra={
                "n_modalities": len(valid_mods),
                "reference_modality": ref_mod,
                "dtw_distance_matrix": dtw_distances,
                "warping_paths": warping_paths,
                "stored_in_obsm": "X_temporal_aligned",
            },
        )

        logg.info(f"DTW 对齐完成: ref={ref_mod}, {len(dtw_distances)} 对模态")
        return mdata

    def _pairwise_dtw(
        self, X_a: np.ndarray, X_b: np.ndarray,
        window_type: str, window_size: float, metric: str,
    ) -> tuple[float, list]:
        """计算两个矩阵之间的 DTW 距离（多变量 DTW）。"""
        try:
            from tslearn.metrics import dtw_path as ts_dtw_path

            max_len = max(X_a.shape[0], X_b.shape[0])
            gc = None
            sc_radius = None
            if window_type == "sakoechiba":
                gc = "sakoe_chiba"
                sc_radius = max(1, int(window_size * max_len))
            elif window_type == "itakura":
                gc = "itakura"

            # 多变量 DTW：使用完整的特征向量
            n_feat = min(X_a.shape[1], X_b.shape[1])
            X_a_mv = X_a[:, :n_feat].copy()
            X_b_mv = X_b[:, :n_feat].copy()

            # P2-17: 加权 DTW — 按权重缩放时间点
            if self.weights is not None:
                w = np.asarray(self.weights, dtype=float)
                w = w / np.sqrt(np.mean(w ** 2))  # 归一化以保持 scale
                if len(w) == X_a_mv.shape[0]:
                    X_a_mv = X_a_mv * w[:, np.newaxis]
                if len(w) == X_b_mv.shape[0]:
                    X_b_mv = X_b_mv * w[:, np.newaxis]

            kwargs = {"global_constraint": gc}
            if sc_radius is not None:
                kwargs["sakoe_chiba_radius"] = sc_radius

            path, dist = ts_dtw_path(X_a_mv, X_b_mv, **kwargs)
            return float(dist), [(int(p[0]), int(p[1])) for p in path]

        except ImportError:
            logg.warning("tslearn 未安装，使用 scipy fallback")
            return self._fallback_dtw(X_a, X_b)

    def _reorder_by_warping(
        self, X: np.ndarray, path: list[tuple[int, int]],
    ) -> np.ndarray:
        """使用 DTW 规整路径重新排序数据（保持 n_obs 不变）。

        对于每个时间索引，根据规整路径中的映射关系对样本重新排序，
        使时间动态更接近参考模态。
        """
        n_obs, n_features = X.shape
        X_aligned = X.copy()

        # 构建映射：source index → 对应的 target index
        src_to_tgt = {}
        for src_idx, tgt_idx in path:
            if 0 <= src_idx < n_obs:
                src_to_tgt[src_idx] = tgt_idx

        if not src_to_tgt:
            return X_aligned

        # 按 target index 排序 source 样本
        sorted_pairs = sorted(src_to_tgt.items(), key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in sorted_pairs]

        # 将排序后的索引映射回数据
        for new_pos, old_idx in enumerate(sorted_indices):
            if new_pos < n_obs and old_idx < n_obs:
                X_aligned[new_pos] = X[old_idx]

        return X_aligned

    def _fallback_dtw(self, X_a: np.ndarray, X_b: np.ndarray) -> tuple[float, list]:
        """使用 scipy 的简化 DTW fallback"""
        from scipy.spatial.distance import cdist

        seq_a = np.mean(X_a, axis=1) if X_a.ndim > 1 else X_a
        seq_b = np.mean(X_b, axis=1) if X_b.ndim > 1 else X_b

        n, m = len(seq_a), len(seq_b)
        cost = cdist(seq_a.reshape(-1, 1), seq_b.reshape(-1, 1), metric="euclidean")

        # 加权 DTW：如果提供了权重
        if self.weights is not None:
            w = np.asarray(self.weights)
            if len(w) == n:
                cost = cost * w[:, np.newaxis]
            elif len(w) == m:
                cost = cost * w[np.newaxis, :]

        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            diag, up, left = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
            if diag <= up and diag <= left:
                i -= 1; j -= 1
            elif up <= left:
                i -= 1
            else:
                j -= 1

        return float(D[n, m]), path[::-1]

    def _smooth(self, X: np.ndarray) -> np.ndarray:
        """移动平均平滑"""
        if X.shape[0] < self.smooth_window:
            return X
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(X.astype(float), size=self.smooth_window, axis=0)
