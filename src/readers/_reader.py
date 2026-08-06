"""
Module1Reader — 读取 Module 1 标准化流水线输出

处理两种数据布局：
1. combined.h5mu（汇总 MuData，所有模态在一个文件中）
2. 各模态独立的 .h5mu 文件（per-modality subdirs）
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from mudata import MuData, read as read_mudata

import _logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class Module1Reader:
    """读取 Module 1（omics_standardization）的标准化组学数据。

    负责：
    - 读取 combined.h5mu 或各模态独立 .h5mu 文件
    - 校验必需的 .obs 元数据列（time, condition, batch）
    - 提取指定的对齐数据层（默认为 X_corrected）
    - 检测各模态的时间尺度（hour/day）
    - 组装为 MuData 容器供 Module 2 使用
    """

    def __init__(
        self,
        data_dir: str | Path,
        layer: str = "X_corrected",
        time_key: str = "time",
        condition_key: str = "condition",
        batch_key: str = "batch",
        time_unit_key: str = "time_unit",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.layer = layer
        self.time_key = time_key
        self.condition_key = condition_key
        self.batch_key = batch_key
        self.time_unit_key = time_unit_key

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def read_all(self, modalities: list[str] | None = None) -> MuData:
        """读取所有可用模态，组装为 MuData。

        优先读取 combined.h5mu，若不存在则逐个读取各模态独立文件。

        Args:
            modalities: 要读取的模态列表，None 或空列表表示全部。

        Returns:
            包含所有模态 AnnData 的 MuData 容器。
        """
        combined_path = self.data_dir / "combined.h5mu"
        if combined_path.exists():
            logg.info(f"读取汇总文件: {combined_path}")
            return self._read_combined(combined_path, modalities)

        logg.info(f"汇总文件不存在，从独立文件读取: {self.data_dir}")
        return self._read_individual(modalities)

    def validate(self, mdata: MuData) -> dict[str, list[str]]:
        """校验 MuData 中各模态的元数据完整性。

        Returns:
            {"ready": [...], "missing_time": [...], "missing_condition": [...], "missing_batch": [...]}
        """
        result: dict[str, list[str]] = {
            "ready": [],
            "missing_time": [],
            "missing_condition": [],
            "missing_batch": [],
        }
        for mod_name, adata in mdata.mod.items():
            issues = []
            if self.time_key not in adata.obs.columns:
                issues.append("time")
                result["missing_time"].append(mod_name)
            if self.condition_key not in adata.obs.columns:
                issues.append("condition")
                result["missing_condition"].append(mod_name)
            if self.batch_key not in adata.obs.columns:
                issues.append("batch")
                result["missing_batch"].append(mod_name)
            if not issues:
                result["ready"].append(mod_name)
            else:
                logg.warning(f"[{mod_name}] 缺少元数据列: {', '.join(issues)}")
        return result

    def get_alignment_matrix(self, adata: AnnData) -> np.ndarray:
        """从 AnnData 中提取对齐矩阵。

        查找顺序：adata.obsm[layer] → adata.layers[layer] → adata.X

        Returns:
            密集 numpy 数组 (n_obs × n_features)
        """
        # 1. 尝试 obsm
        if self.layer in adata.obsm:
            return np.asarray(adata.obsm[self.layer])

        # 2. 尝试 layers
        if self.layer in adata.layers:
            return self._to_dense(adata.layers[self.layer])

        # 3. 回退到 .X
        logg.warning(
            f"未找到层 '{self.layer}'，回退到 .X",
            deep=f"可用 obsm: {list(adata.obsm.keys())}, layers: {list(adata.layers.keys())}",
        )
        return self._to_dense(adata.X)

    def detect_time_scales(self, mdata: MuData) -> dict[str, dict]:
        """检测各模态的时间尺度和范围。

        Returns:
            {modality_name: {"unit": "hour"|"day"|None, "min": ..., "max": ..., "n_unique": ...}}
        """
        scales = {}
        for mod_name, adata in mdata.mod.items():
            info: dict = {"unit": None, "min": None, "max": None, "n_unique": 0}
            if self.time_key in adata.obs.columns:
                times = adata.obs[self.time_key].values
                info["min"] = float(np.min(times))
                info["max"] = float(np.max(times))
                info["n_unique"] = int(len(np.unique(times)))
            if self.time_unit_key in adata.obs.columns:
                units = adata.obs[self.time_unit_key].unique()
                info["unit"] = str(units[0]) if len(units) > 0 else None
            scales[mod_name] = info
        return scales

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _read_combined(self, path: Path, modalities: list[str] | None) -> MuData:
        """读取 combined.h5mu 并筛选模态"""
        mdata = read_mudata(str(path))
        if modalities:
            available = set(mdata.mod.keys())
            selected = set(modalities)
            missing = selected - available
            if missing:
                logg.warning(f"请求的模态不存在于 combined.h5mu 中: {missing}")
            for mod_name in list(mdata.mod.keys()):
                if mod_name not in selected:
                    del mdata.mod[mod_name]
        logg.info(f"已加载 {len(mdata.mod)} 个模态: {list(mdata.mod.keys())}")
        return mdata

    def _read_individual(self, modalities: list[str] | None) -> MuData:
        """读取各模态独立 .h5mu 文件并组装为 MuData"""
        mdata = MuData({})
        available = self._discover_modalities()

        if modalities:
            available = {k: v for k, v in available.items() if k in modalities}

        for mod_name, file_path in available.items():
            try:
                mod_mdata = read_mudata(str(file_path))
                # 独立文件中 AnnData 在 .mod["data"] 中
                if "data" in mod_mdata.mod:
                    adata = mod_mdata.mod["data"]
                else:
                    # 取第一个模态
                    adata = list(mod_mdata.mod.values())[0]
                mdata.mod[mod_name] = adata
                logg.hint(f"  已加载 [{mod_name}]: {adata.n_obs} obs × {adata.n_vars} vars")
            except Exception as e:
                logg.error(f"读取 [{mod_name}] 失败: {e}")

        logg.info(f"已加载 {len(mdata.mod)} 个模态: {list(mdata.mod.keys())}")
        return mdata

    def _discover_modalities(self) -> dict[str, Path]:
        """扫描 data_dir 下的模态子目录，找到各模态 .h5mu 文件"""
        discovered: dict[str, Path] = {}
        if not self.data_dir.exists():
            logg.error(f"数据目录不存在: {self.data_dir}")
            return discovered

        for item in self.data_dir.iterdir():
            if item.is_dir():
                for f in item.glob("*.h5mu"):
                    discovered[item.name] = f
                    break
        return discovered

    @staticmethod
    def _to_dense(matrix) -> np.ndarray:
        """将稀疏或密集矩阵转为密集 numpy 数组"""
        from scipy import sparse

        if sparse.issparse(matrix):
            return matrix.toarray()
        return np.asarray(matrix)
