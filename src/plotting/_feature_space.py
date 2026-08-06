"""特征空间可视化"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import _logging as logg

if TYPE_CHECKING:
    from mudata import MuData
    from matplotlib.figure import Figure


def plot_distribution_comparison(
    mdata: MuData,
    mod_name: str | None = None,
    layer_key: str = "X_feature_aligned",
    show: bool = True,
) -> Figure | None:
    """绘制对齐前后特征分布对比。

    Args:
        mdata: 已对齐的 MuData
        mod_name: 模态名（None=第一个）
        layer_key: 对齐后矩阵键名
        show: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logg.warning("matplotlib 不可用，跳过绘图")
        return None

    if mod_name is None:
        mod_name = list(mdata.mod.keys())[0]

    if mod_name not in mdata.mod:
        logg.warning(f"模态 {mod_name} 不存在")
        return None

    adata = mdata.mod[mod_name]
    X_before = np.asarray(adata.obsm.get("X_corrected", adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X))
    X_after = np.asarray(adata.obsm.get(layer_key, X_before))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 对齐前均值分布
    axes[0].hist(np.mean(X_before, axis=1), bins=50, alpha=0.7, color="steelblue")
    axes[0].set_title(f"[{mod_name}] Before Alignment\nMean per observation")
    axes[0].set_xlabel("Mean expression")
    axes[0].set_ylabel("Frequency")

    # 对齐后均值分布
    axes[1].hist(np.mean(X_after, axis=1), bins=50, alpha=0.7, color="coral")
    axes[1].set_title(f"[{mod_name}] After Alignment\nMean per observation")
    axes[1].set_xlabel("Mean expression")

    # 方差分布对比
    axes[2].hist(np.var(X_before, axis=0), bins=50, alpha=0.5, label="Before", color="steelblue")
    axes[2].hist(np.var(X_after, axis=0), bins=50, alpha=0.5, label="After", color="coral")
    axes[2].set_title(f"[{mod_name}] Variance Distribution")
    axes[2].set_xlabel("Variance per feature")
    axes[2].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_integrated_embedding(
    mdata: MuData,
    color_by: str = "modality",
    show: bool = True,
) -> Figure | None:
    """绘制集成 UMAP 嵌入。

    Args:
        mdata: 已完成降维的 MuData
        color_by: 着色方式
        show: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logg.warning("matplotlib 不可用，跳过绘图")
        return None

    if "X_umap_integrated" not in mdata.obsm:
        logg.warning("未找到集成 UMAP 嵌入，请先运行 integrated_embedding()")
        return None

    X_umap = mdata.obsm["X_umap_integrated"]

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_by == "modality":
        # 按模态着色（从 obsm 形状反推）
        offset = 0
        for mod_name, adata in mdata.mod.items():
            n_obs = adata.n_obs
            ax.scatter(
                X_umap[offset:offset + n_obs, 0],
                X_umap[offset:offset + n_obs, 1],
                s=5, alpha=0.6, label=mod_name,
            )
            offset += n_obs
    else:
        ax.scatter(X_umap[:, 0], X_umap[:, 1], s=5, alpha=0.6)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Integrated UMAP Embedding")
    ax.legend(markerscale=3, loc="upper right")

    if show:
        plt.show()
    return fig
