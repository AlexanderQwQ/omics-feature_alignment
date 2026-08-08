"""
Settings 配置管理类（scanpy 风格）

读取 config/default.yaml 并暴露为 Python 属性，
可通过环境变量或代码覆盖。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ._verbosity import Verbosity

if TYPE_CHECKING:
    from types import TracebackType


# 默认配置（与 config/default.yaml 保持一致）
_DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "data_dir": "../omics_standardization/data/processed/",
        "modalities": [],
        "alignment_layer": "X_corrected",
        "time_key": "time",
        "condition_key": "condition",
        "batch_key": "batch",
        "time_unit_key": "time_unit",
    },
    "temporal": {
        "method": "auto",
        "time_normalization": {
            "enabled": True,
            "target_unit": "hour",
            "conversion_factors": {"day": 24},
        },
        "dtw": {
            "backend": "tslearn",
            "window_type": "sakoechiba",
            "window_size": 0.1,
            "metric": "euclidean",
            "pre_smoothing": False,
        },
        "interpolation": {
            "method": "spline",
            "n_grid": 100,
            "preserve_original": True,
        },
        "pseudotime": {
            "backend": "scanpy_dpt",
            "n_neighbors": 30,
            "n_dcs": 15,
            "root_cells": None,
        },
        "lag_modeling": {
            "max_lag": 5,
            "method": "pearson",
            "window_size": 3,
        },
    },
    "feature_space": {
        "method": "auto",
        "mnn": {
            "n_neighbors": 15,
            "sigma": 1.0,
            "var_adj": True,
            "cos_norm_in": True,
            "cos_norm_out": True,
        },
        "cca": {
            "n_components": 20,
            "scale": True,
            "regularization": [0.1, 0.1],
        },
        "optimal_transport": {
            "backend": "pot",
            "variant": "fused_gromov_wasserstein",
            "alpha": 0.5,
            "epsilon": 0.01,
            "max_iter": 1000,
            "solver": "pgd",
        },
        "manifold": {
            "method": "spectral",
            "n_components": 20,
        },
    },
    "output": {
        "format": "h5mu",
        "compress": True,
        "save_intermediate": True,
        "export_matrix": True,
        "export_report": True,
        "per_modality_subdir": True,
        "combined_filename": "aligned",
    },
    "storage": {
        "minio": {
            "endpoint": "localhost:9000",
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
            "bucket": "omics-alignment",
            "secure": False,
            "fallback_dir": "data/storage/objects",
            "lifecycle": {
                "enabled": False,
                "expiry_days": 90,
            },
        },
        "relational": {
            "dialect": "sqlite",
            "database": "data/storage/alignment.db",
        },
        "graph": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "neo4j",
            "database": "omics",
            "fallback_dir": "data/storage/graph",
        },
    },
    "logging": {
        "level": "info",
        "to_file": True,
        "log_dir": "logs",
    },
}


class Settings:
    """全局配置管理器

    属性：
        input: 输入配置
        temporal: 时间对齐配置
        feature_space: 特征空间校正配置
        output: 输出配置
    """

    def __init__(self) -> None:
        self._config = _DEFAULT_CONFIG.copy()
        self._verbosity = Verbosity.info
        self._root_logger = logging.getLogger("omics_align")
        self._root_logger.setLevel(logging.INFO)
        self._setup_logger()

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def input(self) -> dict:
        return self._config.get("input", {})

    @property
    def temporal(self) -> dict:
        return self._config.get("temporal", {})

    @property
    def feature_space(self) -> dict:
        return self._config.get("feature_space", {})

    @property
    def storage(self) -> dict:
        return self._config.get("storage", {})

    @property
    def output(self) -> dict:
        return self._config.get("output", {})

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: Verbosity | str | int) -> None:
        if isinstance(value, str):
            value = Verbosity[value]
        elif isinstance(value, int):
            value = Verbosity(value)
        self._verbosity = value
        level_map = {
            Verbosity.error: logging.ERROR,
            Verbosity.warning: logging.WARNING,
            Verbosity.info: logging.INFO,
            Verbosity.hint: logging.DEBUG,
            Verbosity.debug: logging.DEBUG,
        }
        self._root_logger.setLevel(level_map.get(value, logging.INFO))

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    def load_config(self, path: str | Path) -> None:
        """从 YAML 文件加载配置，合并到现有配置上"""
        path = Path(path)
        if not path.exists():
            import warnings
            warnings.warn(f"配置文件 {path} 不存在，使用默认配置", stacklevel=2)
            return

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        self._config = self._deep_merge(self._config, loaded)
        self._apply_logging_config()

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """递归合并两个字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_logging_config(self) -> None:
        """应用日志配置到 root logger"""
        log_cfg = self._config.get("logging", {})
        level_name = log_cfg.get("level", "info").upper()
        self._root_logger.setLevel(getattr(logging, level_name, logging.INFO))

        if log_cfg.get("to_file", True):
            log_dir = Path(log_cfg.get("log_dir", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            if not any(isinstance(h, logging.FileHandler) for h in self._root_logger.handlers):
                from logging import FileHandler
                handler = FileHandler(log_dir / "alignment.log", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                self._root_logger.addHandler(handler)

    def _setup_logger(self) -> None:
        """初始化日志处理器"""
        self._root_logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._root_logger.addHandler(handler)

    def __repr__(self) -> str:
        return f"Settings(verbosity={self._verbosity.name})"
