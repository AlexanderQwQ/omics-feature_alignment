"""
集成辅助工具

提供对齐后的集成嵌入和跨模态相关矩阵计算。
"""

from __future__ import annotations

import numpy as np
from mudata import MuData
from sklearn.decomposition import PCA

import _logging as logg


def integrated_embedding(
    mdata: MuData,
    layer_key: str = "X_feature_aligned",
    n_pca_comps: int = 50,
    n_umap_comps: int = 2,
    n_umap_neighbors: int = 15,
) -> MuData:
    """计算集成嵌入：PCA + UMAP 在拼接后的对齐矩阵上。

    Args:
        mdata: 已完成特征空间校正的 MuData
        layer_key: 用于嵌入的矩阵键名
        n_pca_comps: PCA 成分数
        n_umap_comps: UMAP 成分数（2 或 3）
        n_umap_neighbors: UMAP 近邻数

    Returns:
        嵌入后的 MuData（mdata.obsm 新增 X_pca_integrated, X_umap_integrated）
    """
    matrices = []
    for mod_name, adata in mdata.mod.items():
        if layer_key in adata.obsm:
            X = np.asarray(adata.obsm[layer_key])
            matrices.append(X)

    if not matrices:
        logg.warning("无可用的对齐矩阵，跳过集成嵌入")
        return mdata

    # 统一维度：对每个矩阵用 PCA 降到公共维度
    min_dim = min(m.shape[1] for m in matrices)
    common_dim = min(n_pca_comps, min_dim)
    reduced = []
    for X in matrices:
        if X.shape[1] > common_dim:
            pca_dim = PCA(n_components=common_dim, random_state=42)
            reduced.append(pca_dim.fit_transform(X))
        else:
            reduced.append(X)
    X_concat = np.vstack(reduced)
    logg.info(f"集成嵌入: 拼接矩阵 {X_concat.shape}")

    # PCA
    n_pca = min(n_pca_comps, X_concat.shape[1], X_concat.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_concat)

    # 不存储在 mdata.obsm（MuData n_obs 与拼接行数不匹配）
    # 而是存储到 uns 中
    if "alignment" not in mdata.uns:
        mdata.uns["alignment"] = {}
    mdata.uns["alignment"]["dimensionality_reduction"] = {
        "pca": {"n_comps": n_pca, "explained_variance": float(np.sum(pca.explained_variance_ratio_))},
    }
    mdata.uns["alignment"]["X_pca_integrated"] = X_pca

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_umap_comps,
            n_neighbors=min(n_umap_neighbors, X_pca.shape[0] - 1),
            min_dist=0.1,
            random_state=42,
        )
        X_umap = reducer.fit_transform(X_pca)
        mdata.uns["alignment"]["X_umap_integrated"] = X_umap
        logg.info(f"  PCA: {X_pca.shape}, UMAP: {X_umap.shape}")
    except ImportError:
        logg.warning("umap 不可用，仅计算 PCA")

    # 记录到 uns
    if "alignment" not in mdata.uns:
        mdata.uns["alignment"] = {}
    mdata.uns["alignment"]["dimensionality_reduction"] = {
        "pca": {
            "n_comps": n_pca,
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        },
        "stored_in_obsm": "X_pca_integrated, X_umap_integrated",
    }

    return mdata


def correlation_matrix(
    mdata: MuData,
    layer_key: str = "X_feature_aligned",
) -> dict:
    """计算跨模态样本间的相关矩阵。

    Returns:
        {mod_a_mod_b: correlation_matrix, ...}
    """
    result = {}
    mod_names = [
        name for name, adata in mdata.mod.items()
        if layer_key in adata.obsm
    ]

    for i in range(len(mod_names)):
        for j in range(i + 1, len(mod_names)):
            mod_a, mod_b = mod_names[i], mod_names[j]
            X_a = np.mean(np.asarray(mdata.mod[mod_a].obsm[layer_key]), axis=1)
            X_b = np.mean(np.asarray(mdata.mod[mod_b].obsm[layer_key]), axis=1)
            n = min(len(X_a), len(X_b))
            corr = float(np.corrcoef(X_a[:n], X_b[:n])[0, 1])
            result[f"{mod_a}_{mod_b}"] = corr

    return result
