"""测试特征空间选择器"""

from feature_space import FeatureSpaceSelector


def test_selector_select_with_batch(mock_mdata):
    """有批次信息的模态 → MNN"""
    selector = FeatureSpaceSelector()
    method = selector.select(mock_mdata, batch_key="batch")
    # scrna, proteomics 有 batch
    assert method in ("mnn", "cca", "optimal_transport", "manifold")


def test_selector_run_auto(mock_mdata):
    selector = FeatureSpaceSelector()
    result = selector.run(mock_mdata)
    assert "alignment" in mock_mdata.uns
    assert "feature_space" in mock_mdata.uns["alignment"]


def test_selector_run_explicit(mock_mdata):
    selector = FeatureSpaceSelector()
    result = selector.run(mock_mdata, method="manifold")
    assert "manifold" in mock_mdata.uns["alignment"]["feature_space"]["method"]


def test_selector_run_invalid(mock_mdata):
    selector = FeatureSpaceSelector()
    result = selector.run(mock_mdata, method="invalid")
    # 回退到 manifold
    assert "alignment" in mock_mdata.uns
