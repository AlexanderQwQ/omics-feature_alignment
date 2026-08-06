"""测试评估模块"""

import numpy as np
from evaluation._time_consistency import evaluate_time_consistency
from evaluation._distribution import evaluate_distribution_consistency
from evaluation._cross_modality import evaluate_cross_modality_correlation
from tools._evaluation import run_full_evaluation


def test_time_consistency(mock_mdata):
    """测试时间一致性评估"""
    # 先做简单的时间对齐
    for mod_name, adata in mock_mdata.mod.items():
        adata.obsm["X_temporal_aligned"] = np.asarray(adata.obsm["X_corrected"])
        if "aligned_time" not in adata.obs.columns:
            adata.obs["aligned_time"] = adata.obs["time"]

    result = evaluate_time_consistency(mock_mdata)
    assert "sequence_similarity_score" in result
    assert isinstance(result["sequence_similarity_score"], float)


def test_time_consistency_no_time(mock_mdata_no_time):
    result = evaluate_time_consistency(mock_mdata_no_time)
    assert result["sequence_similarity_score"] == 0.0


def test_distribution_consistency(mock_mdata):
    """测试分布一致性评估"""
    for mod_name, adata in mock_mdata.mod.items():
        adata.obsm["X_feature_aligned"] = np.asarray(adata.obsm["X_corrected"])

    result = evaluate_distribution_consistency(mock_mdata)
    assert "mmd_score" in result


def test_cross_modality(mock_mdata):
    """测试跨模态关联评估"""
    for mod_name, adata in mock_mdata.mod.items():
        adata.obsm["X_feature_aligned"] = np.asarray(adata.obsm["X_corrected"])

    result = evaluate_cross_modality_correlation(mock_mdata)
    assert "mean_pearson_correlation" in result


def test_full_evaluation(mock_mdata):
    """测试完整评估"""
    for mod_name, adata in mock_mdata.mod.items():
        adata.obsm["X_temporal_aligned"] = np.asarray(adata.obsm["X_corrected"])
        adata.obsm["X_feature_aligned"] = np.asarray(adata.obsm["X_corrected"])
        adata.obs["aligned_time"] = adata.obs.get("time", np.arange(adata.n_obs))

    result = run_full_evaluation(mock_mdata)
    assert "overall_score" in result
    assert 0 <= result["overall_score"] <= 1
    assert "evaluation" in mock_mdata.uns["alignment"]
