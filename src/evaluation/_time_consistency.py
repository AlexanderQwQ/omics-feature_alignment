"""
时间一致性评估

评估时间/过程对齐后的数据一致性：
- DTW 距离变化
- 序列相似度提升
- 时间排序一致性
"""

from __future__ import annotations

import numpy as np
from mudata import MuData

import _logging as logg


def evaluate_time_consistency(
    mdata: MuData,
    time_key: str = "aligned_time",
    layer_key: str = "X_temporal_aligned",
) -> dict:
    """评估时间对齐后的一致性（含 before/after 对比）。

    Args:
        mdata: 已完成时间对齐的 MuData（含 .uns["alignment"]["before"] 快照）
    """
    metrics: dict = {
        "dtw_distance_reduction": None,
        "sequence_similarity_score": 0.0,
        "time_ordering_consistency": 0.0,
        "similarity_gain": 0.0,
        "modality_details": {},
    }

    valid_mods = [
        name for name, adata in mdata.mod.items()
        if time_key in adata.obs.columns and layer_key in adata.obsm
    ]

    if len(valid_mods) < 2:
        logg.warning("时间一致性评估需要至少 2 个具备时间信息的模态")
        return metrics

    # 计算各模态的均值时间序列
    mean_series = {}
    for mod_name in valid_mods:
        adata = mdata.mod[mod_name]
        times = adata.obs[time_key].values.astype(float)
        X = np.asarray(adata.obsm[layer_key])
        # 按时间排序并聚合
        sorted_idx = np.argsort(times)
        mean_series[mod_name] = {
            "times": times[sorted_idx],
            "mean": np.mean(X[sorted_idx], axis=1),
        }

    # 成对比较
    pairwise_scores = []
    mod_names = list(mean_series.keys())

    for i in range(len(mod_names)):
        for j in range(i + 1, len(mod_names)):
            mod_a, mod_b = mod_names[i], mod_names[j]
            seq_a = mean_series[mod_a]["mean"]
            seq_b = mean_series[mod_b]["mean"]

            # 序列相似度（Pearson 相关）
            min_len = min(len(seq_a), len(seq_b))
            try:
                corr = float(np.corrcoef(seq_a[:min_len], seq_b[:min_len])[0, 1])
                pairwise_scores.append(abs(corr))
            except Exception:
                pairwise_scores.append(0.0)

            # 时间排序一致性
            metrics["modality_details"][f"{mod_a}_{mod_b}"] = {
                "correlation": pairwise_scores[-1] if pairwise_scores else 0.0,
                "n_points_a": len(seq_a),
                "n_points_b": len(seq_b),
            }

    if pairwise_scores:
        metrics["sequence_similarity_score"] = float(np.mean(pairwise_scores))
    metrics["n_modality_pairs"] = len(pairwise_scores)

    # 时间排序一致性（各模态的时间值是否严格递增）
    ordering_scores = []
    for mod_name in valid_mods:
        times = mdata.mod[mod_name].obs[time_key].values.astype(float)
        is_ordered = np.all(np.diff(times) >= 0)
        ordering_scores.append(1.0 if is_ordered else 0.0)

    metrics["time_ordering_consistency"] = float(np.mean(ordering_scores)) if ordering_scores else 0.0

    # before/after 对比
    before_data = mdata.uns.get("alignment", {}).get("before", {}).get("time_consistency", {})
    before_sim = before_data.get("sequence_similarity_score")
    if before_sim is not None and metrics["sequence_similarity_score"] is not None:
        metrics["similarity_gain"] = round(metrics["sequence_similarity_score"] - before_sim, 4)

    logg.info(
        f"时间一致性: 相似度={metrics['sequence_similarity_score']:.3f}, "
        f"增益={metrics.get('similarity_gain', 0):.3f}"
    )
    return metrics
