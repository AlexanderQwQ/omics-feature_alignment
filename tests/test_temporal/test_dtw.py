"""测试 DTW 对齐器"""

import numpy as np
from temporal import DTWAligner


def test_dtw_init():
    aligner = DTWAligner(window_type="sakoechiba", window_size=0.1)
    assert aligner.window_type == "sakoechiba"
    assert aligner.window_size == 0.1


def test_dtw_run(mock_mdata):
    aligner = DTWAligner(pre_smoothing=False)
    result = aligner.run(mock_mdata, time_key="time")
    assert result is mock_mdata  # 原地修改
    assert "alignment" in mock_mdata.uns
    assert "temporal" in mock_mdata.uns["alignment"]
    assert mock_mdata.uns["alignment"]["temporal"]["method"] == "dtw"


def test_dtw_with_smoothing(mock_mdata):
    aligner = DTWAligner(pre_smoothing=True)
    result = aligner.run(mock_mdata, time_key="time")
    assert "X_temporal_aligned" in mock_mdata.mod["scrna"].obsm


def test_dtw_no_time_key(mock_mdata):
    """DTW 对齐，使用 aligned_time 键（may not exist in all modalities）"""
    aligner = DTWAligner()
    result = aligner.run(mock_mdata, time_key="aligned_time")
    # 可能没有模态有 aligned_time，应该正常返回
    assert result is mock_mdata


def test_dtw_weights(mock_mdata):
    """测试加权 DTW 构造"""
    weights = np.ones(100) * 0.5
    aligner = DTWAligner(weights=weights)
    assert aligner.weights is not None
    assert len(aligner.weights) == 100


def test_dtw_metric_param(mock_mdata):
    """测试 metric 参数保留"""
    aligner = DTWAligner(metric="euclidean")
    assert aligner.metric == "euclidean"
