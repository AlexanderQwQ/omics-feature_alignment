"""
MNNAligner — 互为最近邻批次校正器

适用场景：
- 存在明显实验批次差异或数据来源不一致
- 特征空间中呈现整体偏移
- 样本之间存在跨批次匹配关系

基于 scanpy 的 mutual nearest neighbors (MNN) 方法。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseFeatureAligner

if TYPE_CHECKING:
    from anndata import AnnData


class MNNAligner(BaseFeatureAligner):
    """MNN 批次偏移校正器。

    通过构建跨批次样本之间的匹配关系并估计偏移向量，
    消除由实验条件差异引起的系统性偏移。
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        sigma: float = 1.0,
        var_adj: bool = True,
        cos_norm_in: bool = True,
        cos_norm_out: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.var_adj = var_adj
        self.cos_norm_in = cos_norm_in
        self.cos_norm_out = cos_norm_out
        self._method_name = "MNNAligner"

    def run(
        self,
        mdata: MuData,
        batch_key: str = "batch",
        n_neighbors: int | None = None,
        sigma: float | None = None,
        **kwargs,
    ) -> MuData:
        n_neighbors = n_neighbors or self.n_neighbors
        sigma = sigma if sigma is not None else self.sigma

        # 收集所有批次的样本
        all_batches = set()
        for adata in mdata.mod.values():
            if batch_key in adata.obs.columns:
                all_batches.update(adata.obs[batch_key].unique())

        if len(all_batches) < 2:
            logg.info("MNN: 少于 2 个批次，MNN 不适用，返回原始数据")
            for mod_name, adata in mdata.mod.items():
                X = self._get_feature_matrix(adata)
                adata.obsm["X_feature_aligned"] = X
            return mdata

        logg.info(f"MNN 对齐: {len(mdata.mod)} 个模态, {len(all_batches)} 个批次, k={n_neighbors}")

        mnn_log: dict[str, dict] = {}
        all_offset_vectors: dict[str, dict] = {}

        for mod_name, adata in mdata.mod.items():
            if batch_key not in adata.obs.columns:
                logg.warning(f"[{mod_name}] 缺少 '{batch_key}' 列，跳过 MNN")
                adata.obsm["X_feature_aligned"] = self._get_feature_matrix(adata)
                continue

            try:
                X_corrected, offsets = self._run_mnn_correction(
                    adata, batch_key, n_neighbors, sigma,
                )
                adata.obsm["X_feature_aligned"] = X_corrected
                mnn_log[mod_name] = {
                    "n_obs": adata.n_obs,
                    "n_batches": int(adata.obs[batch_key].nunique()),
                    "feature_dim": X_corrected.shape[1],
                }
                if offsets:
                    all_offset_vectors[mod_name] = offsets
                logg.hint(f"  [{mod_name}]: corrected {adata.n_obs} obs × {X_corrected.shape[1]} dims")

            except Exception as e:
                logg.error(f"[{mod_name}] MNN 校正失败: {e}，使用原始矩阵")
                adata.obsm["X_feature_aligned"] = self._get_feature_matrix(adata)

        extra = {
            "n_batches_total": len(all_batches),
            "mnn_log": mnn_log,
            "stored_in_obsm": "X_feature_aligned",
        }
        if all_offset_vectors:
            extra["mnn_offset_vectors"] = all_offset_vectors

        self._store_trace(
            mdata,
            method="mnn",
            params={
                "n_neighbors": n_neighbors,
                "sigma": sigma,
                "var_adj": self.var_adj,
                "batch_key": batch_key,
            },
            extra=extra,
        )

        logg.info(f"MNN 对齐完成: {len(mnn_log)} 个模态")
        return mdata

    def _run_mnn_correction(
        self,
        adata: AnnData,
        batch_key: str,
        n_neighbors: int,
        sigma: float,
    ) -> tuple[np.ndarray, dict | None]:
        """对单个 AnnData 执行 MNN 批次校正。

        Returns:
            (X_corrected, offset_vectors_dict | None)
        """
        try:
            import scanpy as sc
            import scanpy.external as sce

            adata_tmp = adata.copy()
            X = self._get_feature_matrix(adata_tmp)

            # 创建参考 AnnData
            import anndata
            adata_ref = anndata.AnnData(X=X)
            adata_ref.obs[batch_key] = adata_tmp.obs[batch_key].values
            adata_ref.obs_names = adata_tmp.obs_names

            # 使用 scanpy 的 MNN correct
            sce.pp.mnn_correct(
                adata_ref,
                batch_key=batch_key,
                n_neighbors=n_neighbors,
                sigma=sigma,
                var_adj=self.var_adj,
                cos_norm_in=self.cos_norm_in,
                cos_norm_out=self.cos_norm_out,
            )

            return np.asarray(adata_ref.X), None  # scanpy MNN 不产生显式偏移向量

        except ImportError:
            logg.warning("scanpy.external 不可用，使用批次均值中心化 (batch mean-centering)")
            return self._batch_mean_shift(adata, batch_key)

    def _batch_mean_shift(
        self,
        adata: AnnData,
        batch_key: str,
    ) -> tuple[np.ndarray, dict]:
        """批次均值中心化：按批次减去均值偏移。

        注：这不是真正的 MNN 校正，而是简化的批次效应消除。
        仅在 scanpy MNN 不可用时作为降级方案。

        Returns:
            (X_corrected, offset_vectors_dict)
        """
        X = self._get_feature_matrix(adata)
        batches = adata.obs[batch_key].values
        X_corrected = X.copy()

        global_mean = np.mean(X, axis=0)
        offset_vectors: dict[str, dict] = {}

        for batch in np.unique(batches):
            mask = batches == batch
            batch_mean = np.mean(X[mask], axis=0)
            shift = batch_mean - global_mean
            X_corrected[mask] = X[mask] - shift
            offset_vectors[str(batch)] = {
                "norm": float(np.linalg.norm(shift)),
                "n_samples": int(mask.sum()),
            }

        return X_corrected, offset_vectors
