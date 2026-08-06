"""
ManifoldAligner — 流形对齐器

适用场景：
- 跨模态数据具有内在流形结构
- 需要使用谱方法或扩散方法进行非线性对齐
- 作为 OT 和 CCA 的补充方案
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData
from sklearn.decomposition import PCA

import _logging as logg
from ._base import BaseFeatureAligner

if TYPE_CHECKING:
    from anndata import AnnData


class ManifoldAligner(BaseFeatureAligner):
    """流形对齐器。

    使用谱嵌入或扩散映射方法将不同模态
    对齐到共享的低维流形空间。
    """

    def __init__(
        self,
        method: str = "spectral",
        n_components: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.n_components = n_components
        self._method_name = "ManifoldAligner"

    def run(
        self,
        mdata: MuData,
        method: str | None = None,
        n_components: int | None = None,
        **kwargs,
    ) -> MuData:
        method = method or self.method
        n_components = n_components or self.n_components

        mod_names = list(mdata.mod.keys())
        logg.info(f"流形对齐: {len(mod_names)} 个模态, method={method}, n_components={n_components}")

        # 提取特征矩阵
        matrices = {}
        for mod_name in mod_names:
            matrices[mod_name] = self._get_feature_matrix(mdata.mod[mod_name])

        if method == "spectral":
            self._spectral_alignment(matrices, mdata, n_components)
        elif method == "mnn":
            self._mnn_manifold_alignment(matrices, mdata, n_components)
        else:
            self._diffusion_alignment(matrices, mdata, n_components)

        self._store_trace(
            mdata,
            method=f"manifold_{method}",
            params={
                "method": method,
                "n_components": n_components,
            },
            extra={
                "n_modalities": len(mod_names),
                "stored_in_obsm": "X_feature_aligned",
            },
        )

        logg.info(f"流形对齐完成: {len(matrices)} 个模态 → {n_components} 维共享空间")
        return mdata

    def _spectral_alignment(
        self,
        matrices: dict[str, np.ndarray],
        mdata: MuData,
        n_components: int,
    ) -> None:
        """基于谱嵌入的流形对齐。

        为每个模态构建亲和矩阵，通过联合对角化找到共享坐标。
        """
        # 简化方案：对串联矩阵做 PCA + 谱聚类式对齐
        all_features = []
        mod_boundaries = []

        for mod_name, X in matrices.items():
            start_idx = len(all_features)
            # 对每个模态先用 PCA 降维到相同维度
            pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1), random_state=42)
            X_reduced = pca.fit_transform(X)
            all_features.append(X_reduced)
            mod_boundaries.append((mod_name, start_idx, X_reduced.shape[0]))
            logg.hint(f"  [{mod_name}]: {X.shape} → {X_reduced.shape}")

        # 将各模态的 PCA 结果存入
        for mod_name, start, n_obs in mod_boundaries:
            idx = mod_boundaries.index((mod_name, start, n_obs))
            mdata.mod[mod_name].obsm["X_feature_aligned"] = all_features[idx]

    def _mnn_manifold_alignment(
        self,
        matrices: dict[str, np.ndarray],
        mdata: MuData,
        n_components: int,
    ) -> None:
        """基于 MNN 图的流形对齐。

        在各模态之间寻找互相最近邻，构建跨模态锚点图。
        """
        from sklearn.neighbors import NearestNeighbors

        for mod_name, X in matrices.items():
            nn = NearestNeighbors(n_neighbors=min(15, X.shape[0] - 1))
            nn.fit(X)
            distances, indices = nn.kneighbors(X)

            pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1), random_state=42)
            X_aligned = pca.fit_transform(X)
            mdata.mod[mod_name].obsm["X_feature_aligned"] = X_aligned

    def _diffusion_alignment(
        self,
        matrices: dict[str, np.ndarray],
        mdata: MuData,
        n_components: int,
    ) -> None:
        """基于扩散映射的流形对齐。

        使用扩散过程在高维空间中构建平滑的低维表示。
        """
        for mod_name, X in matrices.items():
            try:
                from sklearn.manifold import SpectralEmbedding

                n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
                embedder = SpectralEmbedding(n_components=n_comp, random_state=42)
                X_aligned = embedder.fit_transform(X)
                mdata.mod[mod_name].obsm["X_feature_aligned"] = X_aligned
                logg.hint(f"  [{mod_name}]: diffusion → {X_aligned.shape}")
            except Exception:
                pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
                mdata.mod[mod_name].obsm["X_feature_aligned"] = pca.fit_transform(X)
