"""
综合评估入口

整合时间一致性、分布一致性、跨模态关联性三维度评估，
生成综合评分。
"""

from __future__ import annotations

import numpy as np
from mudata import MuData

import _logging as logg
from ..evaluation._time_consistency import evaluate_time_consistency
from ..evaluation._distribution import evaluate_distribution_consistency
from ..evaluation._cross_modality import evaluate_cross_modality_correlation


def run_full_evaluation(mdata: MuData) -> dict:
    """运行三维度综合评估。

    Args:
        mdata: 已完成两阶段对齐的 MuData

    Returns:
        综合评估结果字典，包含 overall_score
    """
    logg.info("开始三维度综合评估...")

    # 维度一：时间一致性
    time_metrics = evaluate_time_consistency(mdata)

    # 维度二：特征分布一致性
    distribution_metrics = evaluate_distribution_consistency(mdata)

    # 维度三：跨模态关联性
    cross_modality_metrics = evaluate_cross_modality_correlation(mdata)

    # 综合评分
    scores = []

    # 时间一致性 (0→1)
    if time_metrics.get("sequence_similarity_score") is not None:
        scores.append(time_metrics["sequence_similarity_score"])

    # 分布一致性 (MMD 越低越好，Silhouette 越高越好)
    sil = distribution_metrics.get("silhouette_score")
    if sil is not None:
        # 将 silhouette 归一化到 0→1（通常 −1→1, 越高越好）
        scores.append(max(0, (float(sil) + 1) / 2))

    # 跨模态关联性 (0→1)
    cca = cross_modality_metrics.get("mean_canonical_correlation")
    if cca is not None:
        scores.append(float(cca))

    overall_score = float(np.mean(scores)) if scores else 0.0
    overall_score = round(overall_score, 4)

    # 组装最终报告
    evaluation = {
        "time_consistency": time_metrics,
        "distribution_consistency": distribution_metrics,
        "cross_modality_correlation": cross_modality_metrics,
        "overall_score": overall_score,
    }

    # 写入 mdata.uns
    if "alignment" not in mdata.uns:
        mdata.uns["alignment"] = {}
    mdata.uns["alignment"]["evaluation"] = evaluation

    logg.info(f"综合评估完成: overall_score={overall_score:.4f}")
    return evaluation
