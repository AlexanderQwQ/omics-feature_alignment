"""评估可视化"""

from __future__ import annotations

from typing import TYPE_CHECKING

import _logging as logg

if TYPE_CHECKING:
    from mudata import MuData
    from matplotlib.figure import Figure


def plot_evaluation_summary(
    mdata: MuData,
    show: bool = True,
) -> Figure | None:
    """绘制评估指标汇总雷达图/柱状图。

    Args:
        mdata: 已评估的 MuData
        show: 是否显示图形
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logg.warning("matplotlib 不可用，跳过绘图")
        return None

    evaluation = mdata.uns.get("alignment", {}).get("evaluation", {})
    if not evaluation:
        logg.warning("未找到评估数据")
        return None

    # 提取指标
    labels = []
    values = []

    tc = evaluation.get("time_consistency", {})
    if tc.get("sequence_similarity_score") is not None:
        labels.append("Time Similarity")
        values.append(float(tc["sequence_similarity_score"]))

    dc = evaluation.get("distribution_consistency", {})
    if dc.get("silhouette_score") is not None:
        labels.append("Distribution (Silhouette)")
        values.append(max(0, (float(dc["silhouette_score"]) + 1) / 2))

    cc = evaluation.get("cross_modality_correlation", {})
    if cc.get("mean_canonical_correlation") is not None:
        labels.append("Cross-Modality CCA")
        values.append(float(cc["mean_canonical_correlation"]))

    overall = evaluation.get("overall_score", 0)

    if not values:
        logg.warning("没有可用的评估指标")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 柱状图
    colors = plt.cm.RdYlGn([v / max(values) if max(values) > 0 else 0.5 for v in values])
    axes[0].barh(labels, values, color=colors, edgecolor="gray")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Score")
    axes[0].set_title(f"Evaluation Metrics (Overall: {overall:.3f})")
    for i, v in enumerate(values):
        axes[0].text(v + 0.02, i, f"{v:.3f}", va="center")

    # 雷达图（闭合多边形）
    values_radar = values + [values[0]]
    angles = [i * 2 * np.pi / len(values) for i in range(len(values))]
    angles += [angles[0]]

    import numpy as np
    ax_radar = fig.add_subplot(1, 2, 2, projection="polar")
    ax_radar.plot(angles, values_radar, "o-", linewidth=2, color="steelblue")
    ax_radar.fill(angles, values_radar, alpha=0.25, color="steelblue")
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Alignment Quality Radar")

    plt.tight_layout()
    if show:
        plt.show()
    return fig
