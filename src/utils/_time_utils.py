"""
时间尺度归一化工具

处理不同模态之间的时间尺度差异（小时 vs 天），
构建统一时间网格，提供时间标签生成。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData

import _logging as logg

if TYPE_CHECKING:
    from anndata import AnnData

# 默认转换因子
_DEFAULT_CONVERSIONS = {"day": 24, "hour": 1, "minute": 1 / 60, "week": 168}


def normalize_time_scales(
    mdata: MuData,
    target_unit: str = "hour",
    time_key: str = "time",
    time_unit_key: str = "time_unit",
    conversions: dict[str, float] | None = None,
) -> MuData:
    """将各模态的时间值归一化到统一时间单位。

    读取 .obs[time_unit_key] 确定原始单位，
    乘以转换因子得到目标单位的时间值，
    写入 .obs["aligned_time"]。

    Args:
        mdata: 输入 MuData
        target_unit: 目标时间单位（默认 hour）
        time_key: .obs 中的时间列名
        time_unit_key: .obs 中的时间单位列名
        conversions: 自定义转换因子（覆盖默认值）

    Returns:
        修改后的 MuData（原地修改）
    """
    factors = {**_DEFAULT_CONVERSIONS, **(conversions or {})}
    if target_unit not in factors:
        raise ValueError(f"不支持的目标时间单位: {target_unit}，可用: {list(factors.keys())}")

    target_factor = factors[target_unit]
    conversion_log: dict[str, dict] = {}

    for mod_name, adata in mdata.mod.items():
        if time_key not in adata.obs.columns:
            logg.warning(f"[{mod_name}] 缺少时间列 '{time_key}'，跳过时间归一化")
            continue

        times = adata.obs[time_key].values.astype(float)
        original_unit = "hour"  # 默认

        if time_unit_key in adata.obs.columns:
            units = adata.obs[time_unit_key].unique()
            if len(units) > 0:
                original_unit = str(units[0])

        if original_unit in factors:
            scale = target_factor / factors[original_unit]
        else:
            logg.warning(f"[{mod_name}] 未知时间单位 '{original_unit}'，假设为 hour")
            scale = 1.0

        aligned_times = times * scale
        adata.obs["aligned_time"] = aligned_times
        adata.obs["aligned_time"] = adata.obs["aligned_time"].astype(float)

        conversion_log[mod_name] = {
            "original_unit": original_unit,
            "target_unit": target_unit,
            "scale_factor": scale,
            "time_range": [float(np.min(aligned_times)), float(np.max(aligned_times))],
        }

    # 记录到 mdata.uns
    if "alignment" not in mdata.uns:
        mdata.uns["alignment"] = {}
    mdata.uns["alignment"]["temporal"] = {
        **(mdata.uns["alignment"].get("temporal", {})),
        "time_normalization": {
            "applied": True,
            "target_unit": target_unit,
            "conversions": conversion_log,
        },
    }

    logg.info(f"时间归一化完成: {len(conversion_log)} 个模态 → {target_unit}")
    return mdata


def build_common_time_grid(
    mdata: MuData,
    time_key: str = "aligned_time",
    n_grid: int = 100,
    method: str = "union",
) -> np.ndarray:
    """构建跨模态的统一时间网格。

    Args:
        mdata: 已做时间归一化的 MuData
        time_key: 时间列名（默认用归一化后的 'aligned_time'）
        n_grid: 网格点数（method='uniform' 时有效）
        method: 'union' — 合并所有模态的独有时间点
                'uniform' — 在全局最小-最大范围内均匀分布

    Returns:
        排序后的统一时间网格 (numpy 数组)
    """
    if method == "union":
        all_times = []
        for adata in mdata.mod.values():
            if time_key in adata.obs.columns:
                all_times.extend(adata.obs[time_key].values.tolist())
        grid = np.unique(all_times)
    elif method == "uniform":
        all_min, all_max = np.inf, -np.inf
        for adata in mdata.mod.values():
            if time_key in adata.obs.columns:
                t = adata.obs[time_key].values
                all_min = min(all_min, np.min(t))
                all_max = max(all_max, np.max(t))
        grid = np.linspace(all_min, all_max, n_grid)
    else:
        raise ValueError(f"不支持的网格构建方法: {method}，可用: union, uniform")

    return np.sort(grid)


def get_time_label(mdata: MuData, modality: str, time_key: str = "time") -> str:
    """获取某模态的时间标签，包含原始单位和数值范围信息。

    Returns:
        例如 "hour: 0-48" 或 "day: 0-30"
    """
    if modality not in mdata.mod:
        return "unknown"
    adata = mdata.mod[modality]
    unit = "unknown"
    if "time_unit" in adata.obs.columns:
        units = adata.obs["time_unit"].unique()
        if len(units) > 0:
            unit = str(units[0])
    if time_key in adata.obs.columns:
        t = adata.obs[time_key].values
        return f"{unit}: {np.min(t):.0f}-{np.max(t):.0f}"
    return f"{unit}: no data"
