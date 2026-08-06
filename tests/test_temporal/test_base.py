"""测试时间对齐基类"""

import numpy as np
from temporal._base import BaseTemporalAligner


class _DummyAligner(BaseTemporalAligner):
    """用于测试基类功能的虚拟对齐器"""
    def run(self, mdata, time_key="time", **kwargs):
        self._store_trace(mdata, method="dummy", params={"test": True})
        return mdata


def test_base_temporal_validate(mock_mdata):
    aligner = _DummyAligner()
    valid = aligner._validate_time_column(mock_mdata, "time")
    assert "scrna" in valid
    assert "metabolomics" in valid
    assert len(valid) == 3


def test_base_temporal_get_time_series(mock_mdata):
    aligner = _DummyAligner()
    times, X = aligner._get_time_series(mock_mdata.mod["scrna"], "time")
    assert len(times) == 500
    assert X.shape[0] == 500
    # 验证时间排序
    assert np.all(np.diff(times) >= 0)


def test_base_temporal_trace(mock_mdata):
    aligner = _DummyAligner()
    aligner.run(mock_mdata)
    assert "alignment" in mock_mdata.uns
    assert "temporal" in mock_mdata.uns["alignment"]
    assert mock_mdata.uns["alignment"]["temporal"]["method"] == "dummy"


def test_base_temporal_no_time(mock_mdata_no_time):
    aligner = _DummyAligner()
    valid = aligner._validate_time_column(mock_mdata_no_time, "time")
    assert len(valid) == 0
