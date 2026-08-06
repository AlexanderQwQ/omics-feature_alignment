"""时间对齐可视化"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import _logging as logg

if TYPE_CHECKING:
    from mudata import MuData
    from matplotlib.figure import Figure


def plot_warping_paths(
    mdata: MuData,
    pair_key: str | None = None,
    show: bool = True,
) -> Figure | None:
    """绘制 DTW 规整路径。

    Args:
        mdata: 已完成 DTW 对齐的 MuData
        pair_key: 模态对键名（如 "scrna_atac"），None=绘制第一对
        show: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logg.warning("matplotlib 不可用，跳过绘图")
        return None

    temporal_info = mdata.uns.get("alignment", {}).get("temporal", {})
    warping_paths = temporal_info.get("warping_paths", {})

    if not warping_paths:
        logg.warning("未找到规整路径数据")
        return None

    if pair_key is None:
        pair_key = list(warping_paths.keys())[0]

    path = warping_paths.get(pair_key, [])
    if not path:
        logg.warning(f"未找到 '{pair_key}' 的规整路径")
        return None

    xs, ys = zip(*path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, ys, "b-", alpha=0.6, linewidth=0.5)
    ax.plot([0, max(xs)], [0, max(ys)], "r--", alpha=0.3, label="Identity")
    ax.set_xlabel(f"{pair_key.split('_')[0]} (time index)")
    ax.set_ylabel(f"{pair_key.split('_')[-1]} (time index)")
    ax.set_title(f"DTW Warping Path: {pair_key}")
    ax.legend()

    if show:
        plt.show()
    return fig


def plot_time_series_comparison(
    mdata: MuData,
    mod_a: str,
    mod_b: str,
    layer_key: str = "X_temporal_aligned",
    show: bool = True,
) -> Figure | None:
    """绘制两个模态的时间序列对比。

    Args:
        mdata: 已对齐的 MuData
        mod_a, mod_b: 要对比的模态名
        layer_key: 包含时间序列的 obsm 键名
        show: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logg.warning("matplotlib 不可用，跳过绘图")
        return None

    if mod_a not in mdata.mod or mod_b not in mdata.mod:
        logg.warning(f"模态 {mod_a} 或 {mod_b} 不存在")
        return None

    adata_a = mdata.mod[mod_a]
    adata_b = mdata.mod[mod_b]

    if layer_key not in adata_a.obsm or layer_key not in adata_b.obsm:
        logg.warning(f"缺少 '{layer_key}'")
        return None

    times_a = adata_a.obs.get("aligned_time", adata_a.obs.get("time"))
    times_b = adata_b.obs.get("aligned_time", adata_b.obs.get("time"))

    mean_a = np.mean(np.asarray(adata_a.obsm[layer_key]), axis=1)
    mean_b = np.mean(np.asarray(adata_b.obsm[layer_key]), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始
    axes[0].plot(times_a, mean_a, "o-", label=mod_a, markersize=4)
    axes[0].plot(times_b, mean_b, "s-", label=mod_b, markersize=4)
    axes[0].set_title("Mean Expression over Time")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mean Expression")
    axes[0].legend()

    # 标准化后对比
    from scipy import stats
    z_a = stats.zscore(mean_a)
    z_b = stats.zscore(mean_b)
    axes[1].plot(times_a, z_a, "o-", label=f"{mod_a} (z-score)", markersize=4)
    axes[1].plot(times_b, z_b, "s-", label=f"{mod_b} (z-score)", markersize=4)
    axes[1].set_title("Z-Score Normalized Comparison")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Z-Score")
    axes[1].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig
