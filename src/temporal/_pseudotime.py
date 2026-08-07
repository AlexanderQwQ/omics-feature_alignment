"""
PseudotimeAligner — 伪时间排序与阶段映射对齐器

适用场景：
- 不同宿主/实验条件下生物过程不同步
- 免疫应答、感染响应、发育过程
- 单细胞数据或动态过程研究

基于 scanpy DPT（Diffusion Pseudotime）重建高维轨迹。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseTemporalAligner

if TYPE_CHECKING:
    from anndata import AnnData


class PseudotimeAligner(BaseTemporalAligner):
    """伪时间排序与阶段映射对齐器。

    不依赖实验采样时间，而是基于特征变化轨迹
    在高维特征空间中重建动态发展路径。

    使用 scanpy 的 Diffusion Pseudotime (DPT) 算法。
    """

    def __init__(
        self,
        n_neighbors: int = 30,
        n_dcs: int = 15,
        root_cells: int | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.n_dcs = n_dcs
        self.root_cells = root_cells
        self._method_name = "PseudotimeAligner"

    def run(
        self,
        mdata: MuData,
        time_key: str | None = None,
        n_neighbors: int | None = None,
        n_dcs: int | None = None,
        root_cells: int | str | None = None,
        n_stages: int = 5,
        **kwargs,
    ) -> MuData:
        """执行伪时间排序与阶段映射。

        Args:
            mdata: 输入 MuData
            time_key: 忽略（伪时间不依赖实验时间）
            n_neighbors: 近邻数
            n_dcs: 扩散成分数
            root_cells: 根细胞（None=自动检测）
            n_stages: 阶段锚点数量（用于跨模态阶段映射）
        """
        n_neighbors = n_neighbors or self.n_neighbors
        n_dcs = n_dcs or self.n_dcs
        root_cells = root_cells if root_cells is not None else self.root_cells

        logg.info(f"伪时间排序: {len(mdata.mod)} 个模态, stages={n_stages}")

        pseudotime_log: dict[str, dict] = {}
        stage_anchors: dict[str, dict] = {}

        for mod_name, adata in mdata.mod.items():
            try:
                pt = self._compute_pseudotime(adata, n_neighbors, n_dcs, root_cells)
                adata.obs["pseudotime"] = pt

                # 阶段映射：将伪时间等分为 n_stages 个阶段
                stage_boundaries = np.linspace(pt.min(), pt.max(), n_stages + 1)
                stage_labels = np.digitize(pt, stage_boundaries[1:-1])
                adata.obs["process_stage"] = stage_labels.astype(int)

                # 计算阶段锚点（每个阶段的质心）
                anchors = {}
                for s in range(n_stages):
                    mask = stage_labels == s
                    if mask.sum() > 0:
                        if "X_corrected" in adata.obsm:
                            centroid = np.mean(np.asarray(adata.obsm["X_corrected"])[mask], axis=0)
                        else:
                            centroid = np.mean(adata.X[mask].toarray() if hasattr(adata.X, "toarray") else adata.X[mask], axis=0)
                        anchors[f"stage_{s}"] = {
                            "n_cells": int(mask.sum()),
                            "centroid_norm": float(np.linalg.norm(centroid)),
                        }
                stage_anchors[mod_name] = anchors

                # 按伪时间排序
                pt_sorted = np.argsort(pt)
                if "X_corrected" in adata.obsm:
                    X = np.asarray(adata.obsm["X_corrected"])
                elif "X_temporal_aligned" in adata.obsm:
                    X = np.asarray(adata.obsm["X_temporal_aligned"])
                else:
                    X = self._get_time_series(adata, "time")[1]

                adata.obsm["X_temporal_aligned"] = X[pt_sorted]
                adata.obs["aligned_time"] = np.arange(len(pt_sorted)).astype(float)

                pseudotime_log[mod_name] = {
                    "n_cells": adata.n_obs,
                    "n_stages": n_stages,
                    "pseudotime_range": [float(np.min(pt)), float(np.max(pt))],
                }
                logg.hint(f"  [{mod_name}]: {n_stages} 阶段, 伪时间 [{pseudotime_log[mod_name]['pseudotime_range'][0]:.3f}, {pseudotime_log[mod_name]['pseudotime_range'][1]:.3f}]")

            except Exception as e:
                logg.error(f"[{mod_name}] 伪时间计算失败: {e}")
                continue

        self._store_trace(
            mdata,
            method="pseudotime",
            params={
                "n_neighbors": n_neighbors,
                "n_dcs": n_dcs,
                "root_cells": str(root_cells),
            },
            extra={
                "pseudotime_log": pseudotime_log,
                "stage_anchors": stage_anchors,
                "n_stages": n_stages,
                "stored_in_obsm": "X_temporal_aligned",
                "stored_in_obs": "pseudotime, aligned_time, process_stage",
            },
        )

        logg.info(f"伪时间排序完成: {len(pseudotime_log)} 个模态")
        return mdata

    def _compute_pseudotime(
        self,
        adata: AnnData,
        n_neighbors: int,
        n_dcs: int,
        root_cells: int | str | None,
    ) -> np.ndarray:
        """使用 scanpy DPT 计算伪时间。

        Returns:
            伪时间数组 (n_obs,)
        """
        try:
            import scanpy as sc

            # 拷贝 AnnData 以避免修改原始数据
            adata_tmp = adata.copy()

            # 确保有 PCA
            if "X_pca" not in adata_tmp.obsm:
                sc.pp.pca(adata_tmp, n_comps=min(n_dcs * 2, adata_tmp.n_vars, adata_tmp.n_obs - 1))

            # 计算邻居图
            sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, n_pcs=n_dcs)

            # 扩散图
            sc.tl.diffmap(adata_tmp, n_comps=n_dcs)

            # 确定根细胞
            if root_cells is None:
                root_idx = 0  # 自动选第一个细胞作为根
            elif isinstance(root_cells, str) and root_cells in adata_tmp.obs.columns:
                root_idx = int(np.argmax(adata_tmp.obs[root_cells].values))
            else:
                root_idx = int(root_cells)

            # DPT
            adata_tmp.uns["iroot"] = root_idx
            sc.tl.dpt(adata_tmp, n_dcs=n_dcs)

            return adata_tmp.obs["dpt_pseudotime"].values.copy()

        except ImportError:
            logg.warning("scanpy 不可用，使用简化伪时间（PCA 第一主成分）")
            return self._fallback_pseudotime(adata)

    def _fallback_pseudotime(self, adata: AnnData) -> np.ndarray:
        """使用 PCA 第一主成分作为简化伪时间"""
        from sklearn.decomposition import PCA

        if "X_corrected" in adata.obsm:
            X = np.asarray(adata.obsm["X_corrected"])
        else:
            X = np.asarray(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

        pca = PCA(n_components=min(5, X.shape[1], X.shape[0] - 1), random_state=42)
        X_pca = pca.fit_transform(X)
        return X_pca[:, 0]
