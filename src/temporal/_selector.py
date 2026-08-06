"""
TemporalSelector — 时间对齐方法自动选择器

根据数据特征自动选择最优的时间/过程对齐方法：
- DTW: 采样点 ≥5, 时间连续, 局部错位
- Interpolation: 采样点 <5 或时间点不统一
- Pseudotime: 单细胞数据, 无统一采样时间
- Lag: 跨模态存在响应滞后
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg
from ._base import BaseTemporalAligner
from ._dtw import DTWAligner
from ._interpolation import InterpolationAligner
from ._pseudotime import PseudotimeAligner
from ._lag_modeling import LagModelingAligner

if TYPE_CHECKING:
    pass


class TemporalSelector:
    """自动选择并执行时间/过程对齐方法。

    选择启发式：
    - 采样点 ≥5, 时间连续 → DTW
    - 采样点 <5 或时间点不统一 → Interpolation
    - 单细胞数据 → Pseudotime
    - 跨模态响应滞后 → Lag Modeling
    - 多模态混合 → 自动组合 DTW+Interpolation 或 Pseudotime+Lag
    """

    def __init__(self) -> None:
        self._available = {
            "dtw": DTWAligner,
            "interpolation": InterpolationAligner,
            "pseudotime": PseudotimeAligner,
            "lag": LagModelingAligner,
        }

    def select(self, mdata: MuData, time_key: str = "time") -> str:
        """检查数据特征，返回推荐的方法名。

        Returns:
            "dtw" | "interpolation" | "pseudotime" | "lag" | "dtw+interpolation"
        """
        n_time_modalities = 0
        total_time_points = 0
        has_single_cell = False
        has_large_n_obs = False

        for mod_name, adata in mdata.mod.items():
            if time_key in adata.obs.columns:
                n_time_modalities += 1
                n_unique = len(np.unique(adata.obs[time_key].values))
                total_time_points += n_unique

            # 检测单细胞数据（观测数 > 200）
            if adata.n_obs > 200:
                has_single_cell = True
                has_large_n_obs = True

        # 决策逻辑
        if has_single_cell and n_time_modalities == 0:
            logg.info("TemporalSelector: 选择 pseudotime（单细胞数据, 无统一采样时间）")
            return "pseudotime"

        if n_time_modalities >= 2:
            avg_points = total_time_points / n_time_modalities if n_time_modalities > 0 else 0
            if avg_points >= 5:
                logg.info(f"TemporalSelector: 选择 dtw（每模态平均 {avg_points:.1f} 个时间点）")
                return "dtw"
            else:
                logg.info(f"TemporalSelector: 选择 interpolation（每模态平均 {avg_points:.1f} 个时间点）")
                return "interpolation"

        if n_time_modalities >= 1:
            logg.info("TemporalSelector: 选择 interpolation")
            return "interpolation"

        logg.info("TemporalSelector: 回退到 pseudotime")
        return "pseudotime"

    def run(
        self,
        mdata: MuData,
        method: str | None = None,
        time_key: str = "time",
        **kwargs,
    ) -> MuData:
        """自动选择并执行时间对齐。

        Args:
            mdata: 输入 MuData
            method: 显式指定方法名（None=自动选择）
            time_key: 时间列名
            **kwargs: 传递给具体对齐器的参数

        Returns:
            对齐后的 MuData
        """
        # 若指定了 lag 方法，同时执行时间对齐 + 滞后分析
        if method == "lag":
            method_name = self.select(mdata, time_key)
            if method_name != "lag":
                # 先执行基础时间对齐
                primary = self._available.get(method_name, InterpolationAligner)()
                mdata = primary.run(mdata, time_key=time_key, **kwargs)

            # 再执行滞后分析
            lag_aligner = LagModelingAligner(**kwargs.get("lag_modeling", {}))
            mdata = lag_aligner.run(mdata, time_key="aligned_time", **kwargs)
            return mdata

        if method is None:
            method = self.select(mdata, time_key)

        if method not in self._available:
            logg.warning(f"不支持的方法 '{method}'，回退到 interpolation")
            method = "interpolation"

        aligner_cls = self._available[method]
        aligner = aligner_cls(**kwargs.get(method, {}))
        return aligner.run(mdata, time_key=time_key, **kwargs)
