"""
CCAAligner — 典型相关分析对齐器

将不同模态映射到共享典型相关空间，使异构特征空间的数据
在统一低维语义空间中表达。

使用 sklearn CCA + pyrcca (rCCA) 实现。
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

    将不同模态的数据映射到典型相关空间，
    使共享成分最大化跨模态线性相关。
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
        self.regularization = regularization or [0.01, 0.01]
        self._method_name = "CCAAligner"
        self._cca_models: dict[str, object] = {}  # 保存 CCA 模型用于变换

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

        # P1-7: 选择参考模态（特征维度最大的），构建统一共享空间
        ref_mod = max(mod_names, key=lambda m: matrices[m].shape[1])

        # 每个非参考模态分别与参考模态做 CCA
        # 参考模态使用 PCA 作为共享空间（避免多对 CCA 导致的 n_obs 不一致）
        cca_results: dict[str, dict] = {}
        aligned_projections: dict[str, np.ndarray] = {}

        # 参考模态用 PCA 降维到共享空间维度
        from sklearn.decomposition import PCA
        X_ref = matrices[ref_mod]
        n_comp_ref = min(n_components, X_ref.shape[1], X_ref.shape[0] - 1)
        n_comp_ref = max(1, n_comp_ref)
        pca = PCA(n_components=n_comp_ref, random_state=42)
        aligned_projections[ref_mod] = pca.fit_transform(X_ref)

        for mod_name in mod_names:
            if mod_name == ref_mod:
                continue

            X_b = matrices[mod_name]
            n_samples = min(X_ref.shape[0], X_b.shape[0])
            max_comps = min(n_components, n_comp_ref, X_b.shape[1], n_samples - 1)
            max_comps = max(1, max_comps)

            try:
                result = self._fit_cca(
                    X_ref[:n_samples], X_b[:n_samples],
                    max_comps, scale, regularization,
                )
                pair_key = f"{ref_mod}_{mod_name}"
                cca_results[pair_key] = result
                logg.hint(
                    f"  {pair_key}: {max_comps} 成分, "
                    f"平均典型相关={result['mean_correlation']:.4f}, "
                    f"rCCA={result.get('regularized', False)}"
                )

                # 存储非参考模态的 CCA 投影
                aligned_projections[mod_name] = result["X_b_transformed"]

            except Exception as e:
                logg.error(f"  {ref_mod}-{mod_name} CCA 失败: {e}")

        # 将 CCA 变换写入各模态
        for mod_name in mod_names:
            adata = mdata.mod[mod_name]
            if mod_name in aligned_projections:
                proj = aligned_projections[mod_name]
                adata.obsm["X_feature_aligned"] = proj
            else:
                # 没有成功对齐的模态：用 PCA 降维作为后备
                X = matrices[mod_name]
                pca = PCA(
                    n_components=min(n_components, X.shape[1], X.shape[0] - 1),
                    random_state=42,
                )
                adata.obsm["X_feature_aligned"] = pca.fit_transform(X)

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

        logg.info(f"CCA 对齐完成: {len(cca_results)} 对模态, "
                  f"{len(aligned_projections)} 个模态进入共享空间")
        return mdata

    def _fit_cca(
        self, X_a: np.ndarray, X_b: np.ndarray,
        n_components: int, scale: bool, regularization: list[float],
    ) -> dict:
        """拟合 CCA/rCCA 并返回变换后的坐标和相关系数"""
        if scale:
            X_a = StandardScaler().fit_transform(X_a)
            X_b = StandardScaler().fit_transform(X_b)

        regularized = False

        # 尝试 rCCA（pyrcca）
        try:
            import rcca

            cca = rcca.CCA(
                reg=regularization[0],
                numCC=n_components,
                verbose=False,
            )
            cca.train([X_a, X_b])
            # pyrcca: test() returns correlations
            test_result = cca.validate([X_a, X_b])
            canonical_corrs = np.asarray(test_result) if test_result is not None else np.zeros(n_components)
            regularized = True

            # 使用 rCCA 权重直接计算变换后的坐标
            # rCCA 的 compute_weights 返回的是投影后的数据
            outputs = cca.compute_weights([X_a, X_b])
            X_a_c = np.asarray(outputs[0]) if outputs[0] is not None else np.zeros((X_a.shape[0], n_components))
            X_b_c = np.asarray(outputs[1]) if outputs[1] is not None else np.zeros((X_b.shape[0], n_components))

            # 确保维度正确
            if X_a_c.shape[1] != n_components:
                X_a_c = X_a_c[:, :n_components]
            if X_b_c.shape[1] != n_components:
                X_b_c = X_b_c[:, :n_components]

            corrs = np.array([
                np.corrcoef(X_a_c[:, k], X_b_c[:, k])[0, 1]
                for k in range(min(X_a_c.shape[1], X_b_c.shape[1]))
            ])

            return {
                "n_components": n_components,
                "canonical_correlations": corrs.tolist(),
                "mean_correlation": float(np.mean(np.abs(corrs))) if len(corrs) > 0 else 0.0,
                "regularized": True,
                "regularization_weights_applied": True,
                "X_a_transformed": X_a_c,
                "X_b_transformed": X_b_c,
            }

        except ImportError:
            logg.hint("  rcca 未安装，使用 sklearn CCA")

        # sklearn CCA（主实现或 fallback）
        cca = CCA(n_components=n_components, scale=False, max_iter=1000)
        X_a_c, X_b_c = cca.fit_transform(X_a, X_b)

        corrs = np.array([
            np.corrcoef(X_a_c[:, k], X_b_c[:, k])[0, 1]
            for k in range(X_a_c.shape[1])
        ])

        return {
            "n_components": n_components,
            "canonical_correlations": corrs.tolist(),
            "mean_correlation": float(np.mean(np.abs(corrs))),
            "regularized": regularized,
            "X_a_transformed": X_a_c,
            "X_b_transformed": X_b_c,
        }
