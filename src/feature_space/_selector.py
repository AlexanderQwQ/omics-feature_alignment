"""
FeatureSpaceSelector — 特征空间校正方法自动选择器

根据数据特征自动选择最优的特征空间校正方法：
- MNN: 有明确批次标签, 样本存在跨批次匹配
- CCA: 不同模态, 特征空间异构, 样本有对应
- OT: 不同模态, 样本无一一对应, 整体分布差异
- Manifold: 补充方案 (非线性流形结构)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseFeatureAligner
from ._mnn import MNNAligner
from ._cca import CCAAligner
from ._optimal_transport import OptimalTransportAligner
from ._manifold import ManifoldAligner

if TYPE_CHECKING:
    pass


class FeatureSpaceSelector:
    """自动选择并执行特征空间校正方法。

    选择启发式：
    - 有批次标签, n_batches ≥ 2, 样本跨批次可匹配 → MNN
    - 特征空间异构 (不同维度), 样本有对应 → CCA
    - 样本无对应, 纯分布对齐 → OT
    - 非线性关系 → Manifold
    """

    def __init__(self) -> None:
        self._available = {
            "mnn": MNNAligner,
            "cca": CCAAligner,
            "optimal_transport": OptimalTransportAligner,
            "manifold": ManifoldAligner,
        }

    def select(self, mdata: MuData, batch_key: str = "batch") -> str:
        """检查数据特征，返回推荐的方法名。"""
        n_modalities = len(mdata.mod)

        # 检查批次信息
        has_batch = False
        n_batches_total = 0
        for adata in mdata.mod.values():
            if batch_key in adata.obs.columns:
                has_batch = True
                n_batches = adata.obs[batch_key].nunique()
                n_batches_total = max(n_batches_total, n_batches)

        # 检查特征空间异构性（P1-9: 使用变异系数 CV 替代原始方差）
        dims = []
        for adata in mdata.mod.values():
            if "X_temporal_aligned" in adata.obsm:
                dims.append(adata.obsm["X_temporal_aligned"].shape[1])
            elif "X_corrected" in adata.obsm:
                dims.append(adata.obsm["X_corrected"].shape[1])
            else:
                dims.append(adata.n_vars)

        dims_arr = np.array(dims)
        dim_cv = float(np.std(dims_arr) / (np.mean(dims_arr) + 1)) if len(dims) > 1 else 0.0
        heterogeneous = dim_cv > 0.3  # 变异系数 > 0.3 = 异构特征空间

        # 检查样本对应关系
        n_obs_list = [adata.n_obs for adata in mdata.mod.values()]
        balanced_obs = max(n_obs_list) / (min(n_obs_list) + 1) < 3  # 样本量差异 < 3倍

        # 决策
        if has_batch and n_batches_total >= 2:
            logg.info(f"FeatureSpaceSelector: 选择 MNN ({n_batches_total} 个批次)")
            return "mnn"

        if heterogeneous and balanced_obs:
            logg.info(f"FeatureSpaceSelector: 选择 CCA (异构空间 CV={dim_cv:.2f}, 样本均衡)")
            return "cca"

        if heterogeneous and not balanced_obs:
            logg.info(f"FeatureSpaceSelector: 选择 optimal_transport (异构空间, 样本不均衡)")
            return "optimal_transport"

        if n_modalities >= 3:
            logg.info("FeatureSpaceSelector: 选择 optimal_transport (多模态)")
            return "optimal_transport"

        logg.info("FeatureSpaceSelector: 选择 manifold")
        return "manifold"

    def run(
        self,
        mdata: MuData,
        method: str | None = None,
        batch_key: str = "batch",
        **kwargs,
    ) -> MuData:
        """自动选择并执行特征空间校正。

        Args:
            mdata: 输入 MuData（通常已完成时间对齐）
            method: 显式指定方法名（None=自动选择）
            batch_key: 批次列名
            **kwargs: 传递给具体校正器的参数

        Returns:
            校正后的 MuData
        """
        if method is None:
            method = self.select(mdata, batch_key)

        if method not in self._available:
            logg.warning(f"不支持的方法 '{method}'，回退到 manifold")
            method = "manifold"

        aligner_cls = self._available[method]
        aligner = aligner_cls(**kwargs.get(method, {}))
        return aligner.run(mdata, batch_key=batch_key, **kwargs)
