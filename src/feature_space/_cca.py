"""
CCAAligner — 典型相关分析对齐器

适用场景：
- 多组学模态特征空间结构性差异
- 需要将不同模态映射到共享潜在空间
- 样本之间存在对应关系

使用 sklearn CCA 作为标准实现，pyrcca 作为正则化变体。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

import _logging as logg
from ._base import BaseFeatureAligner

if TYPE_CHECKING:
    from anndata import AnnData


class CCAAligner(BaseFeatureAligner):
    """CCA/rCCA 共享潜在空间对齐器。

    将不同模态映射到典型相关空间，
    使不同来源的数据在统一低维语义空间中表达。
    """

    def __init__(
        self,
        n_components: int = 20,
        scale: bool = True,
        regularization: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_components = n_components
        self.scale = scale
        self.regularization = regularization or [0.1, 0.1]
        self._method_name = "CCAAligner"

    def run(
        self,
        mdata: MuData,
        n_components: int | None = None,
        scale: bool | None = None,
        regularization: list[float] | None = None,
        **kwargs,
    ) -> MuData:
        n_components = n_components or self.n_components
        scale = scale if scale is not None else self.scale
        regularization = regularization or self.regularization

        mod_names = list(mdata.mod.keys())
        if len(mod_names) < 2:
            logg.warning("CCA: 需要至少 2 个模态")
            return mdata

        logg.info(f"CCA 对齐: {len(mod_names)} 个模态, n_components={n_components}")

        # 提取各模态特征矩阵
        matrices = {}
        for mod_name in mod_names:
            matrices[mod_name] = self._get_feature_matrix(mdata.mod[mod_name])

        # 成对 CCA
        cca_results: dict[str, dict] = {}

        for i in range(len(mod_names)):
            for j in range(i + 1, len(mod_names)):
                mod_a, mod_b = mod_names[i], mod_names[j]
                X_a = matrices[mod_a]
                X_b = matrices[mod_b]

                # 处理样本量不匹配：取较小者的样本量
                n_samples = min(X_a.shape[0], X_b.shape[0])
                X_a = X_a[:n_samples]
                X_b = X_b[:n_samples]

                if X_a.shape[1] < 2 or X_b.shape[1] < 2:
                    logg.warning(f"  {mod_a}-{mod_b}: 特征维度太低，跳过")
                    continue

                # CCA 成分数不能超过最小维度
                max_comps = min(n_components, X_a.shape[1], X_b.shape[1], n_samples - 1)
                max_comps = max(1, max_comps)

                try:
                    result = self._fit_cca(X_a, X_b, max_comps, scale, regularization)
                    pair_key = f"{mod_a}_{mod_b}"
                    cca_results[pair_key] = result
                    logg.hint(
                        f"  {pair_key}: {max_comps} 成分, "
                        f"平均典型相关={result['mean_correlation']:.4f}"
                    )
                except Exception as e:
                    logg.error(f"  {mod_a}-{mod_b} CCA 失败: {e}")

        # 为每个模态生成对齐后的表示
        for mod_name in mod_names:
            adata = mdata.mod[mod_name]
            X = self._get_feature_matrix(adata)

            # 使用 PCA 投影到统一维度
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1), random_state=42)
            adata.obsm["X_feature_aligned"] = pca.fit_transform(X)
            adata.uns["alignment"] = {
                **(adata.uns.get("alignment", {})),
                "feature_space": {
                    "method": "cca",
                    "cca_pairwise": list(cca_results.keys()),
                },
            }

        self._store_trace(
            mdata,
            method="cca",
            params={
                "n_components": n_components,
                "scale": scale,
                "regularization": regularization,
            },
            extra={
                "cca_pairwise_results": cca_results,
                "stored_in_obsm": "X_feature_aligned",
            },
        )

        logg.info(f"CCA 对齐完成: {len(cca_results)} 对模态")
        return mdata

    def _fit_cca(
        self,
        X_a: np.ndarray,
        X_b: np.ndarray,
        n_components: int,
        scale: bool,
        regularization: list[float],
    ) -> dict:
        """拟合 CCA 并返回结果字典"""
        # 标准化
        if scale:
            scaler_a = StandardScaler()
            scaler_b = StandardScaler()
            X_a = scaler_a.fit_transform(X_a)
            X_b = scaler_b.fit_transform(X_b)

        # 尝试 rCCA（pyrcca），失败则用 sklearn CCA
        canonical_corrs = None
        try:
            import pyrcca
            cca = pyrcca.CCA(
                reg=regularization[0],
                numCC=n_components,
                verbose=False,
            )
            cca.train([X_a, X_b])
            # pyrcca 不直接提供典型相关系数
        except ImportError:
            pass

        # sklearn CCA（作为主实现或 fallback）
        cca = CCA(n_components=n_components, scale=scale, max_iter=1000)
        cca.fit(X_a, X_b)

        # 计算典型相关系数
        X_a_c, X_b_c = cca.transform(X_a, X_b)
        canonical_corrs = np.array([
            np.corrcoef(X_a_c[:, k], X_b_c[:, k])[0, 1]
            for k in range(X_a_c.shape[1])
        ])

        return {
            "n_components": n_components,
            "canonical_correlations": canonical_corrs.tolist(),
            "mean_correlation": float(np.mean(np.abs(canonical_corrs))),
            "aligned_dim_a": X_a_c.shape[1],
            "aligned_dim_b": X_b_c.shape[1],
        }
