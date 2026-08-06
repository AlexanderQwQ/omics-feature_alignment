"""
da 命名空间 — 动态对齐用户 API

模仿 scanpy/muon 的 pp 命名空间风格，
提供便捷的时间对齐和特征空间校正入口。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import _logging as logg
from temporal import DTWAligner, InterpolationAligner, PseudotimeAligner, LagModelingAligner, TemporalSelector
from feature_space import MNNAligner, CCAAligner, OptimalTransportAligner, ManifoldAligner, FeatureSpaceSelector

if TYPE_CHECKING:
    from mudata import MuData


# =============================================================================
# 阶段一：时间/过程对齐
# =============================================================================


def temporal_dtw(
    mdata: MuData,
    time_key: str = "aligned_time",
    window_type: str = "sakoechiba",
    window_size: float = 0.1,
    **kwargs,
) -> MuData:
    """DTW 动态时间规整。

    Args:
        mdata: 输入 MuData
        time_key: 时间列名
        window_type: 窗口约束类型 (sakoechiba | itakura | none)
        window_size: 窗口大小比例 (0.0-1.0)
    """
    aligner = DTWAligner(window_type=window_type, window_size=window_size, **kwargs)
    return aligner.run(mdata, time_key=time_key)


def temporal_interpolation(
    mdata: MuData,
    time_key: str = "aligned_time",
    method: str = "spline",
    n_grid: int = 100,
    **kwargs,
) -> MuData:
    """时间插值与统一映射。

    Args:
        mdata: 输入 MuData
        time_key: 时间列名
        method: 插值方法 (linear | spline | loess)
        n_grid: 统一时间网格点数
    """
    aligner = InterpolationAligner(method=method, n_grid=n_grid, **kwargs)
    return aligner.run(mdata, time_key=time_key)


def temporal_pseudotime(
    mdata: MuData,
    n_neighbors: int = 30,
    n_dcs: int = 15,
    root_cells: int | str | None = None,
    **kwargs,
) -> MuData:
    """伪时间排序与阶段映射。

    Args:
        mdata: 输入 MuData
        n_neighbors: 近邻数
        n_dcs: 扩散成分数
        root_cells: 根细胞（None=自动检测）
    """
    aligner = PseudotimeAligner(
        n_neighbors=n_neighbors, n_dcs=n_dcs, root_cells=root_cells, **kwargs,
    )
    return aligner.run(mdata)


def temporal_lag(
    mdata: MuData,
    time_key: str = "aligned_time",
    max_lag: int = 5,
    method: str = "pearson",
    **kwargs,
) -> MuData:
    """时间延迟建模与相关性对齐。

    Args:
        mdata: 输入 MuData
        time_key: 时间列名
        max_lag: 最大滞后步数
        method: 相关方法 (pearson | spearman | mutual_info)
    """
    aligner = LagModelingAligner(max_lag=max_lag, method=method, **kwargs)
    return aligner.run(mdata, time_key=time_key)


def temporal(
    mdata: MuData,
    method: str | None = None,
    time_key: str = "time",
    **kwargs,
) -> MuData:
    """自动选择时间对齐方法。

    Args:
        mdata: 输入 MuData
        method: 显式指定方法（None=自动选择）
              auto | dtw | interpolation | pseudotime | lag
        time_key: 时间列名
    """
    selector = TemporalSelector()
    return selector.run(mdata, method=method, time_key=time_key, **kwargs)


# =============================================================================
# 阶段二：特征空间校正
# =============================================================================


def feature_mnn(
    mdata: MuData,
    batch_key: str = "batch",
    n_neighbors: int = 15,
    sigma: float = 1.0,
    **kwargs,
) -> MuData:
    """MNN 互为最近邻批次校正。

    Args:
        mdata: 输入 MuData
        batch_key: 批次列名
        n_neighbors: 近邻数
        sigma: 高斯核带宽
    """
    aligner = MNNAligner(n_neighbors=n_neighbors, sigma=sigma, **kwargs)
    return aligner.run(mdata, batch_key=batch_key)


def feature_cca(
    mdata: MuData,
    n_components: int = 20,
    scale: bool = True,
    **kwargs,
) -> MuData:
    """CCA / rCCA 共享潜在空间对齐。

    Args:
        mdata: 输入 MuData
        n_components: 典型成分数
        scale: 是否标准化
    """
    aligner = CCAAligner(n_components=n_components, scale=scale, **kwargs)
    return aligner.run(mdata)


def feature_ot(
    mdata: MuData,
    variant: str = "fused_gromov_wasserstein",
    alpha: float = 0.5,
    epsilon: float = 0.01,
    **kwargs,
) -> MuData:
    """最优传输分布对齐。

    Args:
        mdata: 输入 MuData
        variant: GW 变体 (gw | fused_gw | coot | unbalanced)
        alpha: FGW 权衡参数（0=pure GW, 1=pure Wasserstein）
        epsilon: 熵正则化系数
    """
    aligner = OptimalTransportAligner(
        variant=variant, alpha=alpha, epsilon=epsilon, **kwargs,
    )
    return aligner.run(mdata)


def feature_space(
    mdata: MuData,
    method: str | None = None,
    batch_key: str = "batch",
    **kwargs,
) -> MuData:
    """自动选择特征空间校正方法。

    Args:
        mdata: 输入 MuData
        method: 显式指定方法（None=自动选择）
              auto | mnn | cca | optimal_transport | manifold
        batch_key: 批次列名
    """
    selector = FeatureSpaceSelector()
    return selector.run(mdata, method=method, batch_key=batch_key, **kwargs)


# =============================================================================
# 一键式
# =============================================================================


def align(
    mdata: MuData,
    temporal_method: str | None = None,
    feature_space_method: str | None = None,
    time_key: str = "time",
    batch_key: str = "batch",
    **kwargs,
) -> MuData:
    """运行完整两阶段对齐。

    先时间/过程对齐，再特征空间校正。

    Args:
        mdata: 输入 MuData
        temporal_method: 时间对齐方法（None=自动）
        feature_space_method: 特征空间方法（None=自动）
        time_key: 时间列名
        batch_key: 批次列名

    Returns:
        对齐后的 MuData
    """
    logg.info("=== 阶段一：动态时间与过程对齐 ===")
    mdata = temporal(mdata, method=temporal_method, time_key=time_key, **kwargs)

    logg.info("=== 阶段二：特征空间补充校正 ===")
    mdata = feature_space(mdata, method=feature_space_method, batch_key=batch_key, **kwargs)

    logg.info("完整两阶段对齐完成")
    return mdata
