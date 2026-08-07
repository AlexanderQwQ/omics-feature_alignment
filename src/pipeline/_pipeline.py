"""
DynamicAlignmentPipeline — 动态特征对齐流水线

编排两阶段对齐流程：
1. 读取 Module 1 输出
2. 阶段一：动态时间与过程对齐
3. 阶段二：特征空间补充校正
4. 三维度评估
5. 保存对齐结果
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import _logging as logg
from _settings import settings
from readers import Module1Reader
from temporal import TemporalSelector
from feature_space import FeatureSpaceSelector
from tools._evaluation import run_full_evaluation
from tools._integration import integrated_embedding
from tools._export import export_to_csv, export_report
from utils._time_utils import normalize_time_scales

if TYPE_CHECKING:
    from mudata import MuData


class DynamicAlignmentPipeline:
    """端到端多组学动态特征对齐流水线。

    步骤：
        1. 读取 Module 1 处理后数据
        2. 时间尺度归一化（day → hour）
        3. 阶段一：动态时间/过程对齐
        4. 阶段二：特征空间补充校正
        5. 集成降维（PCA + UMAP）
        6. 三维度评估
        7. 保存对齐结果 + 导出 CSV/报告

    Usage:
        pipeline = DynamicAlignmentPipeline("config/default.yaml")
        mdata = pipeline.run()
        # 或传入已有 MuData
        mdata = pipeline.run(data=mdata)
    """

    def __init__(self, config: str | Path | None = None) -> None:
        if config is not None:
            settings.load_config(Path(config))
        self._steps: list[str] = []
        self._results: dict[str, Any] = {}
        self._start_time = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        data: MuData | None = None,
        modalities: list[str] | None = None,
    ) -> MuData:
        """运行完整对齐流水线。

        Args:
            input_path: Module 1 处理后数据目录（None=使用配置文件中的路径）
            output_path: 输出目录（None=当前目录下的 data/aligned）
            data: 预加载的 MuData（跳过加载步骤）
            modalities: 要处理的模态子集（None=全部）

        Returns:
            对齐后的 MuData
        """
        logg.info("=" * 60)
        logg.info("DynamicAlignmentPipeline 开始运行")
        logg.info("=" * 60)

        # Step 0: 加载数据
        if data is not None:
            mdata = data
            logg.info("使用预加载的 MuData")
        else:
            input_cfg = settings.input
            data_dir = input_path or input_cfg.get("data_dir", "data/processed/")
            mdata = self._step_load(Path(data_dir), modalities)

        # 记录输入来源
        mdata.uns["alignment"] = {
            "pipeline_version": "0.1.0",
            "input_source": str(input_path or settings.input.get("data_dir")),
            "modalities_processed": list(mdata.mod.keys()),
            "steps_executed": [],
            "timestamp": str(self._start_time),
        }

        # Step 1: 时间尺度归一化
        logg.info("--- 时间尺度归一化 ---")
        mdata = normalize_time_scales(mdata)
        self._steps.append("time_normalization")

        # ★ 对齐前快照 — 用于 before/after 对比
        logg.info("--- 对齐前基线快照 ---")
        mdata = self._snapshot_before(mdata)

        # Step 2: 阶段一 — 时间/过程对齐
        logg.info("--- 阶段一：动态时间与过程对齐 ---")
        mdata = self._step_temporal(mdata)
        self._steps.append("temporal")
        mdata.uns["alignment"]["steps_executed"].append("temporal")

        # Step 3: 阶段二 — 特征空间校正
        logg.info("--- 阶段二：特征空间补充校正 ---")
        mdata = self._step_feature_space(mdata)
        self._steps.append("feature_space")
        mdata.uns["alignment"]["steps_executed"].append("feature_space")

        # Step 4: 集成降维
        logg.info("--- 集成降维 ---")
        mdata = integrated_embedding(mdata)
        self._steps.append("dimensionality_reduction")
        mdata.uns["alignment"]["steps_executed"].append("dimensionality_reduction")

        # Step 5: 评估（使用 before/after 快照计算差值）
        logg.info("--- 三维度评估 ---")
        self._results["evaluation"] = run_full_evaluation(mdata)
        self._steps.append("evaluation")

        # Step 6: 保存
        logg.info("--- 保存结果 ---")
        output_dir = Path(output_path or "data/aligned")
        self._save(mdata, output_dir)

        elapsed = datetime.now(timezone.utc) - self._start_time
        logg.info(f"Pipeline 完成: {self._steps}, 耗时 {elapsed}")
        return mdata

    # ------------------------------------------------------------------
    # 步骤方法
    # ------------------------------------------------------------------

    def _step_load(self, data_dir: Path, modalities: list[str] | None) -> MuData:
        """加载 Module 1 输出数据"""
        config = settings.input
        reader = Module1Reader(
            data_dir=data_dir,
            layer=config.get("alignment_layer", "X_corrected"),
            time_key=config.get("time_key", "time"),
            condition_key=config.get("condition_key", "condition"),
            batch_key=config.get("batch_key", "batch"),
            time_unit_key=config.get("time_unit_key", "time_unit"),
        )
        mdata = reader.read_all(modalities)

        # 校验
        validation = reader.validate(mdata)
        if validation["ready"]:
            logg.info(f"已就绪模态: {validation['ready']}")

        # 检测时间尺度
        scales = reader.detect_time_scales(mdata)
        for mod_name, info in scales.items():
            logg.hint(f"  [{mod_name}] 时间: unit={info['unit']}, range=[{info['min']}, {info['max']}], n={info['n_unique']}")

        return mdata

    def _step_temporal(self, mdata: MuData) -> MuData:
        """阶段一：时间/过程对齐"""
        config = settings.temporal
        method = config.get("method", "auto")
        if method == "auto":
            method = None

        selector = TemporalSelector()
        return selector.run(mdata, method=method, time_key="time")

    def _step_feature_space(self, mdata: MuData) -> MuData:
        """阶段二：特征空间校正"""
        config = settings.feature_space
        method = config.get("method", "auto")
        if method == "auto":
            method = None

        selector = FeatureSpaceSelector()
        return selector.run(mdata, method=method, batch_key="batch")

    def _snapshot_before(self, mdata: MuData) -> MuData:
        """在对齐前保存基线快照，供 before/after 对比评估使用。"""
        from evaluation._cross_modality import evaluate_cross_modality_correlation
        from evaluation._distribution import evaluate_distribution_consistency
        from evaluation._time_consistency import evaluate_time_consistency

        # 临时使用未对齐的数据做基线评估
        for mod_name, adata in mdata.mod.items():
            if "X_corrected" in adata.obsm:
                adata.obsm["X_temporal_aligned"] = np.asarray(adata.obsm["X_corrected"])
                adata.obsm["X_feature_aligned"] = np.asarray(adata.obsm["X_corrected"])

        time_before = evaluate_time_consistency(mdata)
        dist_before = evaluate_distribution_consistency(mdata)
        cross_before = evaluate_cross_modality_correlation(mdata)

        mdata.uns["alignment"]["before"] = {
            "time_consistency": time_before,
            "distribution_consistency": dist_before,
            "cross_modality_correlation": cross_before,
        }
        logg.hint(
            f"基线快照: time_seq={time_before.get('sequence_similarity_score', 0):.3f}, "
            f"cross_corr={cross_before.get('mean_pearson_correlation', 0):.3f}"
        )
        return mdata

    def _save(self, mdata: MuData, output_dir: Path) -> None:
        """保存对齐结果"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整 MuData
        filename = settings.output.get("combined_filename", "aligned")
        output_file = output_dir / f"{filename}.h5mu"
        mdata.write(str(output_file))
        logg.info(f"已保存: {output_file}")

        # 导出 CSV
        if settings.output.get("export_matrix", True):
            csv_dir = output_dir / "csv"
            export_to_csv(mdata, csv_dir)

        # 导出报告
        if settings.output.get("export_report", True):
            report_path = output_dir / "alignment_report.json"
            export_report(mdata, report_path)

        # 各模态独立文件
        if settings.output.get("per_modality_subdir", True):
            from mudata import MuData
            for mod_name, adata in mdata.mod.items():
                mod_dir = output_dir / mod_name
                mod_dir.mkdir(parents=True, exist_ok=True)
                mod_mdata = MuData({"data": adata})
                mod_mdata.write(str(mod_dir / f"{mod_name}_aligned.h5mu"))
