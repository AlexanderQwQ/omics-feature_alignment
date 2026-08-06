"""测试时间对齐选择器"""

from temporal import TemporalSelector


def test_selector_select_dtw(mock_mdata):
    """选择器根据时间点特征自动选择方法"""
    selector = TemporalSelector()
    method = selector.select(mock_mdata, time_key="time")
    # 3 个模态含 time，平均约 4.3 时间点 < 5 → interpolation
    # 如果平均 ≥ 5 → dtw
    assert method in ("dtw", "interpolation")


def test_selector_select_no_time(mock_mdata_no_time):
    """无时间列 → pseudotime"""
    selector = TemporalSelector()
    method = selector.select(mock_mdata_no_time, time_key="time")
    assert method == "pseudotime"


def test_selector_run_auto(mock_mdata):
    selector = TemporalSelector()
    result = selector.run(mock_mdata)  # method=None → auto
    assert "alignment" in mock_mdata.uns
    assert "temporal" in mock_mdata.uns["alignment"]


def test_selector_run_explicit(mock_mdata):
    selector = TemporalSelector()
    result = selector.run(mock_mdata, method="interpolation")
    assert mock_mdata.uns["alignment"]["temporal"]["method"] == "interpolation"


def test_selector_run_invalid_method(mock_mdata):
    """无效方法应回退到 interpolation"""
    selector = TemporalSelector()
    result = selector.run(mock_mdata, method="nonexistent_method")
    assert "alignment" in mock_mdata.uns
