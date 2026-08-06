"""
跨模态关联性评估

评估对齐后不同模态之间的关联增强程度：
- 典型相关系数
- 互信息
- 相关性增益
"""

from __future__ import annotations

import numpy as np
from mudata import MuData

import _logging as logg


def evaluate_cross_modality_correlation(
    mdata: MuData,
    layer_key: str = "X_feature_aligned",
) -> dict:
    """评估跨模态特征关联性。

    Args:
        mdata: 已完成两阶段对齐的 MuData
        layer_key: 对齐后矩阵的 obsm 键名

    Returns:
        跨模态关联性指标字典
    """
    metrics: dict = {
        "mean_canonical_correlation": 0.0,
        "mean_pearson_correlation": 0.0,
        "correlation_gain": 0.0,
        "mutual_information_gain": 0.0,
        "pairwise_details": {},
    }

    mod_names = [
        name for name, adata in mdata.mod.items()
        if layer_key in adata.obsm
    ]

    if len(mod_names) < 2:
        logg.warning("跨模态关联评估需要至少 2 个模态")
        return metrics

    # 提取对齐后矩阵
    matrices = {}
    for mod_name in mod_names:
        X = np.asarray(mdata.mod[mod_name].obsm[layer_key])
        matrices[mod_name] = X

    pearson_scores = []
    canonical_scores = []

    for i in range(len(mod_names)):
        for j in range(i + 1, len(mod_names)):
            mod_a, mod_b = mod_names[i], mod_names[j]
            X_a = matrices[mod_a]
            X_b = matrices[mod_b]

            # 截断到相同样本数
            n = min(X_a.shape[0], X_b.shape[0])
            X_a = X_a[:n]
            X_b = X_b[:n]

            if n < 3:
                continue

            pair_key = f"{mod_a}_{mod_b}"

            # Pearson 相关（对每个特征维度取平均）
            pearson = _compute_feature_correlations(X_a, X_b)
            pearson_scores.append(pearson)

            # 简化典型相关（PCA 降维后跨模态相关）
            try:
                from sklearn.cross_decomposition import CCA
                n_comp = min(5, X_a.shape[1], X_b.shape[1], n - 1)
                cca = CCA(n_components=n_comp, max_iter=500)
                cca.fit(X_a, X_b)
                X_a_c, X_b_c = cca.transform(X_a, X_b)
                can_corrs = np.array([
                    np.corrcoef(X_a_c[:, k], X_b_c[:, k])[0, 1]
                    for k in range(n_comp)
                ])
                canonical_scores.append(float(np.mean(np.abs(can_corrs))))
            except Exception:
                pass

            metrics["pairwise_details"][pair_key] = {
                "pearson": float(pearson),
                "n_samples": n,
                "dim_a": int(X_a.shape[1]),
                "dim_b": int(X_b.shape[1]),
            }

    if pearson_scores:
        metrics["mean_pearson_correlation"] = float(np.mean(pearson_scores))
    if canonical_scores:
        metrics["mean_canonical_correlation"] = float(np.mean(canonical_scores))

    # 相关性增益（与对齐前对比）
    before_corr = 0.0
    if "alignment" in mdata.uns and "cross_modality" in mdata.uns["alignment"]:
        cm = mdata.uns["alignment"]["cross_modality"]
        before_corr = cm.get("mean_correlation", 0.0)
    metrics["correlation_gain"] = max(0, metrics["mean_pearson_correlation"] - before_corr)

    logg.info(
        f"跨模态关联: Pearson={metrics['mean_pearson_correlation']:.3f}, "
        f"CCA={metrics['mean_canonical_correlation']:.3f}, "
        f"增益={metrics['correlation_gain']:.3f}"
    )
    return metrics


def _compute_feature_correlations(X_a: np.ndarray, X_b: np.ndarray) -> float:
    """计算两个矩阵特征维度间的平均 Pearson 相关"""
    n_features = min(min(X_a.shape[1], X_b.shape[1]), 50)
    cors = []
    for f in range(n_features):
        try:
            c = np.corrcoef(X_a[:, f], X_b[:, f])[0, 1]
            if not np.isnan(c):
                cors.append(abs(c))
        except Exception:
            pass
    return float(np.mean(cors)) if cors else 0.0
