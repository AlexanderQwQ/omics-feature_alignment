"""
输入校验工具

在对齐 Pipeline 运行前验证 MuData 中的数据结构完整性。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import _logging as logg

if TYPE_CHECKING:
    from mudata import MuData


def validate_mdata_for_alignment(
    mdata: MuData,
    time_key: str = "time",
    condition_key: str = "condition",
    batch_key: str = "batch",
    require_time: bool = False,
) -> dict:
    """校验 MuData 是否满足对齐 Pipeline 的最低要求。

    Args:
        mdata: 待校验的 MuData
        time_key: 时间列名
        condition_key: 条件列名
        batch_key: 批次列名
        require_time: 是否强制要求 time 列存在

    Returns:
        校验结果字典: {"valid": bool, "warnings": [...], "errors": [...]}
    """
    result: dict = {"valid": True, "warnings": [], "errors": []}

    if len(mdata.mod) == 0:
        result["valid"] = False
        result["errors"].append("MuData 不包含任何模态")
        return result

    if len(mdata.mod) < 2:
        result["warnings"].append(
            f"只有 {len(mdata.mod)} 个模态，跨模态对齐功能受限"
        )

    for mod_name, adata in mdata.mod.items():
        # 检查基本元数据
        if time_key not in adata.obs.columns:
            msg = f"[{mod_name}] 缺少时间列 '{time_key}'"
            if require_time:
                result["errors"].append(msg)
                result["valid"] = False
            else:
                result["warnings"].append(msg)

        if condition_key not in adata.obs.columns:
            result["warnings"].append(f"[{mod_name}] 缺少条件列 '{condition_key}'")

        if batch_key not in adata.obs.columns:
            result["warnings"].append(f"[{mod_name}] 缺少批次列 '{batch_key}'")

        # 检查数据矩阵
        if adata.n_obs == 0:
            result["errors"].append(f"[{mod_name}] 观测数为 0")
            result["valid"] = False

        if adata.n_vars == 0:
            result["errors"].append(f"[{mod_name}] 变量数为 0")
            result["valid"] = False

        # 检查是否有 NaN
        if hasattr(adata.X, "data"):
            if np.any(np.isnan(adata.X.data)):
                result["warnings"].append(f"[{mod_name}] 数据包含 NaN 值")

    return result


def check_modality_compatibility(
    mdata: MuData,
    mod_a: str,
    mod_b: str,
    alignment_layer: str = "X_corrected",
) -> dict:
    """检查两个模态之间是否可以执行对齐。

    Returns:
        {"compatible": bool, "issues": [...], "suggestion": str}
    """
    result: dict = {"compatible": True, "issues": [], "suggestion": "ready"}
    adata_a = mdata.mod[mod_a]
    adata_b = mdata.mod[mod_b]

    # 检查对齐层是否存在
    if alignment_layer not in adata_a.obsm and alignment_layer not in adata_a.layers:
        result["issues"].append(
            f"[{mod_a}] 缺少对齐层 '{alignment_layer}'，"
            f"可用: obsm={list(adata_a.obsm.keys())}, layers={list(adata_a.layers.keys())}"
        )
        result["suggestion"] = f"回退到 [{mod_a}].X"

    if alignment_layer not in adata_b.obsm and alignment_layer not in adata_b.layers:
        result["issues"].append(
            f"[{mod_b}] 缺少对齐层 '{alignment_layer}'，"
            f"可用: obsm={list(adata_b.obsm.keys())}, layers={list(adata_b.layers.keys())}"
        )
        result["suggestion"] = f"回退到 [{mod_b}].X"

    # 检查维度是否合理
    dim_a = adata_a.obsm.get(alignment_layer, adata_a.X).shape[1]
    dim_b = adata_b.obsm.get(alignment_layer, adata_b.X).shape[1]
    if dim_a < 2 or dim_b < 2:
        result["issues"].append(
            f"特征维度太低: [{mod_a}]={dim_a}, [{mod_b}]={dim_b}"
        )
        result["suggestion"] = "考虑使用原始特征层 (normalized) 而非 X_corrected"

    if result["issues"]:
        result["compatible"] = len(result["issues"]) < 3

    return result
