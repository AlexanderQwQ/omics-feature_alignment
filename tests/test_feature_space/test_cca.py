"""测试 CCA 对齐器"""

from feature_space import CCAAligner


def test_cca_init():
    aligner = CCAAligner(n_components=10, scale=True)
    assert aligner.n_components == 10
    assert aligner.scale is True


def test_cca_run(mock_mdata):
    aligner = CCAAligner(n_components=5)
    result = aligner.run(mock_mdata)
    assert "alignment" in mock_mdata.uns
    assert "feature_space" in mock_mdata.uns["alignment"]
    assert mock_mdata.uns["alignment"]["feature_space"]["method"] == "cca"
    # 验证 X_feature_aligned 存在
    assert "X_feature_aligned" in mock_mdata.mod["scrna"].obsm


def test_cca_single_modality(mock_mdata_no_time, mock_adata_scrna):
    """单模态不应 crash"""
    from mudata import MuData
    mdata = MuData({"scrna": mock_adata_scrna})
    aligner = CCAAligner(n_components=5)
    result = aligner.run(mdata)
    assert "X_feature_aligned" not in mdata.mod["scrna"].obsm
