"""
矩阵处理工具

稀疏/稠密矩阵转换、子采样、维度对齐等。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

import _logging as logg


def to_dense(matrix) -> np.ndarray:
    """将稀疏或密集矩阵统一转为密集 numpy 数组。"""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def subsample_matrix(
    X: np.ndarray,
    n_samples: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """对矩阵行进行随机子采样。

    Args:
        X: 输入矩阵 (n_obs × n_features)
        n_samples: 目标采样数
        random_state: 随机种子

    Returns:
        (X_subsampled, selected_indices)
    """
    n_obs = X.shape[0]
    if n_samples >= n_obs:
        return X.copy(), np.arange(n_obs)

    rng = np.random.RandomState(random_state)
    indices = rng.choice(n_obs, size=n_samples, replace=False)
    indices.sort()
    return X[indices].copy(), indices


def align_matrices(
    matrices: list[np.ndarray],
    axis: int = 1,
    method: str = "pad",
) -> list[np.ndarray]:
    """将多个矩阵对齐到相同维度。

    用于处理不同模态之间特征维度不一致的情况。

    Args:
        matrices: 矩阵列表
        axis: 对齐的轴（0=行, 1=列）
        method: 'pad' — 零填充到最大维度
                'truncate' — 截断到最小维度
                'pca' — PCA 降维到最小维度

    Returns:
        维度对齐后的矩阵列表
    """
    if not matrices:
        return []

    target_dim = _get_target_dim(matrices, axis, method)
    aligned = []

    for i, X in enumerate(matrices):
        current_dim = X.shape[axis]
        if current_dim == target_dim:
            aligned.append(X.copy())
        elif method == "pad" or method == "truncate":
            aligned.append(_resize_matrix(X, target_dim, axis))
        elif method == "pca":
            aligned.append(_pca_reduce(X, target_dim, axis))
        else:
            raise ValueError(f"不支持的对齐方法: {method}")

    return aligned


def _get_target_dim(matrices: list[np.ndarray], axis: int, method: str) -> int:
    dims = [m.shape[axis] for m in matrices]
    if method == "truncate" or method == "pca":
        return min(dims)
    else:
        return max(dims)


def _resize_matrix(X: np.ndarray, target_dim: int, axis: int) -> np.ndarray:
    """通过填充或截断调整矩阵维度"""
    current_dim = X.shape[axis]
    if current_dim == target_dim:
        return X.copy()
    if current_dim > target_dim:
        # 截断
        slc = [slice(None)] * X.ndim
        slc[axis] = slice(0, target_dim)
        return X[tuple(slc)].copy()
    else:
        # 零填充
        pad_shape = list(X.shape)
        pad_shape[axis] = target_dim - current_dim
        padding = np.zeros(pad_shape, dtype=X.dtype)
        return np.concatenate([X, padding], axis=axis)


def _pca_reduce(X: np.ndarray, target_dim: int, axis: int) -> np.ndarray:
    """使用 PCA 降维"""
    from sklearn.decomposition import PCA

    if axis == 0:
        X = X.T

    n_components = min(target_dim, min(X.shape))
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(to_dense(X))

    if axis == 0:
        reduced = reduced.T

    var_explained = pca.explained_variance_ratio_.sum()
    logg.hint(f"PCA 降维: {X.shape[1]} → {n_components}, 解释方差: {var_explained:.3f}")

    return reduced
