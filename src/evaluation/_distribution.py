"""
特征分布一致性评估

评估特征空间校正后的分布对齐效果：
- MMD（最大均值差异）
- Wasserstein 距离
- Silhouette 分数
"""

from __future__ import annotations

import numpy as np
from mudata import MuData

import _logging as logg


def evaluate_distribution_consistency(
    mdata: MuData,
    layer_key: str = "X_feature_aligned",
    batch_key: str = "batch",
) -> dict:
    """评估特征空间校正后的分布一致性。

    Args:
        mdata: 已完成特征空间校正的 MuData
        layer_key: 校正后矩阵的 obsm 键名
        batch_key: 批次列名

    Returns:
        分布一致性指标字典
    """
    metrics: dict = {
        "mmd_score": None,
        "wasserstein_score": None,
        "silhouette_score": None,
        "distribution_shift": {},
        "modality_details": {},
    }

    # 统计对齐前后对比（从 .uns 读取）
    if "alignment" in mdata.uns and "feature_space" in mdata.uns["alignment"]:
        fs_info = mdata.uns["alignment"]["feature_space"]
        metrics["distribution_shift"] = fs_info.get("distribution_shift", {})

    valid_mods = [
        name for name, adata in mdata.mod.items()
        if layer_key in adata.obsm and batch_key in adata.obs.columns
    ]

    if len(valid_mods) < 1:
        logg.warning("分布一致性评估需要至少 1 个具备批次信息的模态")
        return metrics

    mmd_scores = []
    wasserstein_scores = []
    silhouette_scores = []

    for mod_name in valid_mods:
        adata = mdata.mod[mod_name]
        X = np.asarray(adata.obsm[layer_key])
        batches = adata.obs[batch_key].values
        unique_batches = np.unique(batches)

        if len(unique_batches) < 2:
            continue

        # MMD（批次间差异）
        mmd_val = _compute_mmd(X, batches, unique_batches)
        mmd_scores.append(mmd_val)

        # 批间方差 vs 批内方差
        batch_means = []
        for b in unique_batches:
            batch_means.append(np.mean(X[batches == b], axis=0))

        # 简化的 Wasserstein 近似（批间均值距离）
        wass = 0.0
        count = 0
        for i in range(len(batch_means)):
            for j in range(i + 1, len(batch_means)):
                wass += np.linalg.norm(batch_means[i] - batch_means[j])
                count += 1
        if count > 0:
            wasserstein_scores.append(wass / count)

        # Silhouette（批次混合度）
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import LabelEncoder
            labels = LabelEncoder().fit_transform(batches)
            sil = silhouette_score(X[:min(1000, X.shape[0])], labels[:min(1000, X.shape[0])])
            silhouette_scores.append(float(sil))
        except Exception:
            silhouette_scores.append(0.0)

        metrics["modality_details"][mod_name] = {
            "n_batches": int(len(unique_batches)),
            "n_obs": adata.n_obs,
            "feature_dim": X.shape[1],
        }

    if mmd_scores:
        metrics["mmd_score"] = float(np.mean(mmd_scores))
    if wasserstein_scores:
        metrics["wasserstein_score"] = float(np.mean(wasserstein_scores))
    if silhouette_scores:
        metrics["silhouette_score"] = float(np.mean(silhouette_scores))

    # before/after 对比
    before = mdata.uns.get("alignment", {}).get("before", {}).get("distribution_consistency", {})
    before_mmd = before.get("mmd_score")
    if before_mmd is not None and metrics.get("mmd_score") is not None:
        metrics["mmd_reduction"] = round(before_mmd - metrics["mmd_score"], 4)

    logg.info(
        f"分布一致性: MMD={metrics.get('mmd_score', 'N/A')}, "
        f"降幅={metrics.get('mmd_reduction', 'N/A')}"
    )
    return metrics


def _compute_mmd(X: np.ndarray, batches: np.ndarray, unique_batches: np.ndarray) -> float:
    """计算批次间 MMD"""
    n_max = min(500, X.shape[0])
    if n_max < 10:
        return 0.0

    # 简化 MMD：对每对批次计算均值差异
    diffs = []
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            Xi = X[batches == unique_batches[i]]
            Xj = X[batches == unique_batches[j]]
            if len(Xi) > 1 and len(Xj) > 1:
                diff = np.linalg.norm(np.mean(Xi, axis=0) - np.mean(Xj, axis=0))
                diffs.append(diff)

    return float(np.mean(diffs)) if diffs else 0.0
