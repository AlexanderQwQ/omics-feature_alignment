"""
共享测试 Fixtures

模拟 Module 1 输出的 MuData，用于测试 Module 2 各组件。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData
from scipy import sparse


@pytest.fixture
def mock_adata_scrna():
    """模拟 scRNA AnnData（500 obs × 50 vars, X_corrected）"""
    np.random.seed(42)
    n_obs, n_vars = 500, 50
    X = sparse.csr_matrix(np.random.negative_binomial(10, 0.3, (n_obs, n_vars)).astype(np.float32))
    X_corrected = np.random.randn(n_obs, 32).astype(np.float32)
    normalized = np.random.randn(n_obs, n_vars).astype(np.float32)

    adata = AnnData(X=X)
    adata.obsm["X_corrected"] = X_corrected
    adata.layers["normalized"] = normalized
    adata.obs["time"] = np.tile([0, 6, 12, 24, 48], 100).astype(float)
    adata.obs["condition"] = "LPS_stimulation"
    adata.obs["batch"] = np.random.choice(["donor_A", "donor_B"], n_obs)
    adata.obs["time_unit"] = "hour"
    adata.uns["standardization"] = {
        "strategy": {"modality": "scrna"},
        "batch_correction": {"method": "harmony"},
        "normalization": {"method": "scran"},
    }
    return adata


@pytest.fixture
def mock_adata_metabolomics():
    """模拟代谢组 AnnData（80 obs × 30 vars）"""
    np.random.seed(43)
    n_obs, n_vars = 80, 30
    X = sparse.csr_matrix(np.abs(np.random.randn(n_obs, n_vars)).astype(np.float32))
    X_corrected = np.random.randn(n_obs, 16).astype(np.float32)
    normalized = np.random.randn(n_obs, n_vars).astype(np.float32)

    adata = AnnData(X=X)
    adata.obsm["X_corrected"] = X_corrected
    adata.layers["normalized"] = normalized
    adata.obs["time"] = np.tile([0, 4, 12, 48], 20).astype(float)
    adata.obs["condition"] = "high_fat_diet"
    adata.obs["batch"] = "run_01"
    adata.obs["time_unit"] = "hour"
    adata.uns["standardization"] = {
        "strategy": {"modality": "metabolomics"},
        "normalization": {"method": "quantile"},
    }
    return adata


@pytest.fixture
def mock_adata_proteomics():
    """模拟蛋白组 AnnData（100 obs × 20 vars）"""
    np.random.seed(44)
    n_obs, n_vars = 100, 20
    X = sparse.csr_matrix(np.abs(np.random.randn(n_obs, n_vars)).astype(np.float32))
    X_corrected = np.random.randn(n_obs, 16).astype(np.float32)

    adata = AnnData(X=X)
    adata.obsm["X_corrected"] = X_corrected
    adata.obs["time"] = np.tile([0, 8, 24, 72], 25).astype(float)
    adata.obs["condition"] = "cytokine_induction"
    adata.obs["batch"] = np.random.choice(["panel_v1", "panel_v2"], n_obs)
    adata.obs["time_unit"] = "hour"
    adata.uns["standardization"] = {
        "strategy": {"modality": "proteomics"},
    }
    return adata


@pytest.fixture
def mock_adata_microbiome():
    """模拟微生物组 AnnData（30 obs × 25 vars, day 时间单位）"""
    np.random.seed(45)
    n_obs, n_vars = 30, 25
    X = sparse.csr_matrix(np.abs(np.random.randn(n_obs, n_vars)).astype(np.float32))
    X_corrected = np.random.randn(n_obs, 16).astype(np.float32)

    adata = AnnData(X=X)
    adata.obsm["X_corrected"] = X_corrected
    adata.obs["time"] = np.tile([0, 7, 30], 10).astype(float)
    adata.obs["condition"] = "probiotics_intervention"
    adata.obs["batch"] = np.random.choice(["stool", "oral"], n_obs)
    adata.obs["time_unit"] = "day"
    adata.uns["standardization"] = {
        "strategy": {"modality": "microbiome"},
    }
    return adata


@pytest.fixture
def mock_mdata(mock_adata_scrna, mock_adata_metabolomics, mock_adata_proteomics):
    """模拟 Module 1 输出的 MuData（3 模态）"""
    return MuData({
        "scrna": mock_adata_scrna,
        "metabolomics": mock_adata_metabolomics,
        "proteomics": mock_adata_proteomics,
    })


@pytest.fixture
def mock_mdata_with_day(mock_adata_scrna, mock_adata_metabolomics, mock_adata_microbiome):
    """包含两种时间尺度（hour + day）的 MuData"""
    return MuData({
        "scrna": mock_adata_scrna,
        "metabolomics": mock_adata_metabolomics,
        "microbiome": mock_adata_microbiome,
    })


@pytest.fixture
def mock_mdata_no_time():
    """不含时间列的 MuData"""
    adata_a = AnnData(X=np.random.randn(50, 10))
    adata_a.obsm["X_corrected"] = np.random.randn(50, 8)
    adata_a.obs["condition"] = "A"
    adata_a.obs["batch"] = "batch_1"

    adata_b = AnnData(X=np.random.randn(30, 15))
    adata_b.obsm["X_corrected"] = np.random.randn(30, 8)
    adata_b.obs["condition"] = "B"
    adata_b.obs["batch"] = "batch_2"

    return MuData({"mod_a": adata_a, "mod_b": adata_b})
