"""
插值标记管理工具

管理插值/估计点的标记，确保数据可追溯性：
- 原始观测点标记为 False
- 插值/估计点标记为 True
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import _logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


def mark_interpolated(
    adata: AnnData,
    original_mask: np.ndarray,
    column_name: str = "is_interpolated",
) -> AnnData:
    """标记哪些观测是插值生成的。

    Args:
        adata: AnnData 对象
        original_mask: 布尔数组，True 表示原始观测，False 表示插值点
        column_name: 标记列名

    Returns:
        修改后的 AnnData（原地修改）
    """
    if len(original_mask) != adata.n_obs:
        raise ValueError(
            f"mask 长度 ({len(original_mask)}) 与观测数 ({adata.n_obs}) 不匹配"
        )

    # is_interpolated: True = 估计点, False = 原始观测点
    adata.obs[column_name] = ~original_mask.astype(bool)
    adata.obs[column_name] = adata.obs[column_name].astype(bool)

    n_original = int(original_mask.sum())
    n_interpolated = int((~original_mask).sum())
    logg.hint(
        f"插值标记完成: {n_original} 个原始观测点, {n_interpolated} 个插值估计点"
    )

    return adata


def get_interpolation_stats(adata: AnnData, column_name: str = "is_interpolated") -> dict:
    """获取插值统计信息。

    Returns:
        {"n_original": int, "n_interpolated": int, "ratio": float, "time_points": {...}}
    """
    if column_name not in adata.obs.columns:
        return {"n_original": adata.n_obs, "n_interpolated": 0, "ratio": 1.0, "time_points": {}}

    mask = adata.obs[column_name].values
    n_interp = int(mask.sum())
    n_orig = adata.n_obs - n_interp

    stats = {
        "n_original": n_orig,
        "n_interpolated": n_interp,
        "ratio": n_orig / adata.n_obs if adata.n_obs > 0 else 1.0,
        "time_points": {},
    }

    # 分别统计原始和插值的时间范围
    if "aligned_time" in adata.obs.columns:
        orig_times = adata.obs.loc[~mask, "aligned_time"].values if n_orig > 0 else np.array([])
        interp_times = adata.obs.loc[mask, "aligned_time"].values if n_interp > 0 else np.array([])
        stats["time_points"] = {
            "original_range": [float(np.min(orig_times)), float(np.max(orig_times))]
            if len(orig_times) > 0
            else None,
            "interpolated_range": [float(np.min(interp_times)), float(np.max(interp_times))]
            if len(interp_times) > 0
            else None,
        }

    return stats
