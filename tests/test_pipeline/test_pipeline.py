"""测试 DynamicAlignmentPipeline"""

import numpy as np
from pipeline import DynamicAlignmentPipeline


def test_pipeline_init():
    pipeline = DynamicAlignmentPipeline()
    assert pipeline is not None


def test_pipeline_with_data(mock_mdata):
    """测试传入预加载 MuData 的 Pipeline"""
    pipeline = DynamicAlignmentPipeline()
    result = pipeline.run(data=mock_mdata)
    assert result is mock_mdata
    assert "alignment" in mock_mdata.uns
    assert "steps_executed" in mock_mdata.uns["alignment"]
    assert "temporal" in mock_mdata.uns["alignment"]["steps_executed"]
    assert "feature_space" in mock_mdata.uns["alignment"]["steps_executed"]


def test_pipeline_output_keys(mock_mdata):
    """检查输出包含预期的 obsm 键"""
    pipeline = DynamicAlignmentPipeline()
    result = pipeline.run(data=mock_mdata)

    # 时间对齐结果
    for mod_name in mock_mdata.mod:
        assert "aligned_time" in mock_mdata.mod[mod_name].obs.columns, \
            f"{mod_name} 缺少 aligned_time"

    # 特征空间结果
    for mod_name in mock_mdata.mod:
        assert "X_feature_aligned" in mock_mdata.mod[mod_name].obsm, \
            f"{mod_name} 缺少 X_feature_aligned"

    # 集成嵌入存储在 uns 中
    assert "X_pca_integrated" in mock_mdata.uns["alignment"], \
        "缺少 X_pca_integrated"


def test_pipeline_evaluation(mock_mdata):
    """检查评估结果"""
    pipeline = DynamicAlignmentPipeline()
    result = pipeline.run(data=mock_mdata)
    eval_result = mock_mdata.uns["alignment"].get("evaluation", {})
    assert "overall_score" in eval_result
    assert 0 <= eval_result["overall_score"] <= 1


def test_pipeline_single_modality(mock_adata_scrna):
    """测试单模态情况"""
    from mudata import MuData
    mdata = MuData({"scrna": mock_adata_scrna})
    pipeline = DynamicAlignmentPipeline()
    result = pipeline.run(data=mdata)  # 不应 crash
    assert "alignment" in mdata.uns
