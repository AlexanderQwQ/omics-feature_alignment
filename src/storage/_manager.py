"""
StorageManager — 混合存储协调器

统一管理三种存储后端:
    - MinIO / S3: 大型对齐 MuData / 特征矩阵文件
    - SQLite / DM8: 关系型元数据 (实验记录/模态详情/运行记录/指标)
    - Neo4j: 图数据库 (对齐关系/跨模态关联/轨迹)

生命周期:
    store = StorageManager(config_path_or_dict)
    store.connect()                              # 连接所有后端
    store.save_experiment("E001", ...)            # 记录实验
    store.save_alignment_run("run_001", ...)      # 记录流水线运行
    store.put_aligned_mudata("E001", path)        # 上传对齐 MuData
    store.build_alignment_knowledge_graph(mdata)   # 构建知识图谱
    store.disconnect()                            # 断开所有连接

与 Module 1 的 StorageManager 保持接口兼容。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _logging as logg
from _settings import settings

if TYPE_CHECKING:
    from mudata import MuData


class StorageManager:
    """混合存储协调器

    用法:
        store = StorageManager()

        # 方式 1: 自动从 settings 中加载配置
        store.connect()

        # 方式 2: 手动指定配置
        store = StorageManager.from_config("config/default.yaml")
        store.connect()

        with store:  # context manager 自动 connect/disconnect
            store.save_experiment("E001", temporal_method="dtw", feature_space_method="cca")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._minio: Any = None
        self._relational: Any = None
        self._graph: Any = None
        self._connected = False

    @classmethod
    def from_config(cls, path: str | Path) -> StorageManager:
        """从 YAML 配置文件加载存储配置"""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
        storage_config = full_config.get("storage", {})
        return cls(storage_config)

    @classmethod
    def from_settings(cls) -> StorageManager:
        """从全局 settings 对象加载存储配置"""
        try:
            storage_config = getattr(settings, "storage", {})
            return cls(storage_config)
        except Exception:
            return cls()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> StorageManager:
        """连接所有存储后端"""
        from ._minio import MinIOClient
        from ._relational import RelationalDBClient
        from ._graph import GraphDBClient

        # MinIO (对象存储)
        minio_config = self._config.get("minio", {})
        self._minio = MinIOClient(minio_config)
        self._minio.connect()

        # 关系型数据库
        relational_config = self._config.get("relational", {})
        self._relational = RelationalDBClient(relational_config)
        self._relational.connect()

        # 图数据库
        graph_config = self._config.get("graph", {})
        self._graph = GraphDBClient(graph_config)
        self._graph.connect()

        self._connected = True
        logg.info("混合存储管理器已就绪 (MinIO + RelationalDB + GraphDB)")
        return self

    def disconnect(self) -> None:
        """断开所有存储后端"""
        for client in [self._minio, self._relational, self._graph]:
            if client is not None and hasattr(client, "disconnect"):
                client.disconnect()
        self._connected = False
        logg.info("混合存储连接已关闭")

    def __enter__(self) -> StorageManager:
        return self.connect()

    def __exit__(self, *args: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def minio(self) -> Any:
        """对象存储客户端"""
        return self._minio

    @property
    def db(self) -> Any:
        """关系型数据库客户端"""
        return self._relational

    @property
    def graph(self) -> Any:
        """图数据库客户端"""
        return self._graph

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # High-level workflow methods — 对齐实验
    # ------------------------------------------------------------------

    def save_experiment(
        self,
        experiment_id: str,
        *,
        input_source: str | None = None,
        modalities: list[str] | None = None,
        temporal_method: str | None = None,
        feature_space_method: str | None = None,
        time_normalized: bool = False,
        n_total_obs: int | None = None,
        n_total_features: int | None = None,
        pipeline_version: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> str:
        """保存对齐实验记录到关系型数据库

        Returns:
            experiment_id
        """
        if self._relational is None:
            raise RuntimeError("存储管理器未连接，请先调用 connect()")

        return self._relational.save_experiment(
            experiment_id=experiment_id,
            input_source=input_source,
            modalities=modalities,
            temporal_method=temporal_method,
            feature_space_method=feature_space_method,
            time_normalized=time_normalized,
            n_total_obs=n_total_obs,
            n_total_features=n_total_features,
            pipeline_version=pipeline_version,
            config_snapshot=config_snapshot,
            notes=notes,
        )

    def save_modality_detail(
        self,
        experiment_id: str,
        modality_name: str,
        **kwargs: Any,
    ) -> int:
        """保存单个模态的对齐详情"""
        if self._relational is None:
            raise RuntimeError("存储管理器未连接")
        return self._relational.save_modality(experiment_id, modality_name, **kwargs)

    def save_modalities_from_mdata(
        self,
        experiment_id: str,
        mdata: MuData,
    ) -> list[int]:
        """从 MuData 批量保存所有模态的对齐详情

        Args:
            experiment_id: 实验 ID
            mdata: 对齐后的 MuData 对象

        Returns:
            modality_id 列表
        """
        ids: list[int] = []
        time_key = "aligned_time"
        batch_key = "batch"

        for mod_name, adata in mdata.mod.items():
            n_features_before = adata.n_vars
            n_features_after = adata.n_vars
            if "X_feature_aligned" in adata.obsm:
                n_features_after = adata.obsm["X_feature_aligned"].shape[1]

            n_time_points = 0
            time_unit = None
            time_range_min = None
            time_range_max = None
            if time_key in adata.obs.columns:
                times = adata.obs[time_key]
                n_time_points = times.nunique()
                time_range_min = float(times.min())
                time_range_max = float(times.max())
            if "time_unit" in adata.obs.columns:
                time_unit = str(adata.obs["time_unit"].iloc[0]) if adata.n_obs > 0 else None

            n_interpolated = 0
            if "is_interpolated" in adata.obs.columns:
                n_interpolated = int(adata.obs["is_interpolated"].sum())

            n_batches = 0
            if batch_key in adata.obs.columns:
                n_batches = adata.obs[batch_key].nunique()

            mid = self._relational.save_modality(
                experiment_id=experiment_id,
                modality_name=mod_name,
                n_obs=adata.n_obs,
                n_features_before=n_features_before,
                n_features_after=n_features_after,
                n_time_points=n_time_points,
                time_unit=time_unit,
                time_range_min=time_range_min,
                time_range_max=time_range_max,
                n_interpolated=n_interpolated,
                n_original=adata.n_obs - n_interpolated,
                n_batches=n_batches,
            )
            ids.append(mid)

        logg.info(f"已保存 {len(ids)} 个模态详情: experiment={experiment_id}")
        return ids

    # ------------------------------------------------------------------
    # High-level workflow methods — 流水线运行
    # ------------------------------------------------------------------

    def record_alignment_run(
        self,
        experiment_id: str,
        *,
        temporal_method: str | None = None,
        feature_space_method: str | None = None,
        n_modalities_processed: int | None = None,
        steps_executed: list[str] | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """记录一次对齐流水线运行

        Returns:
            run_id (UUID 前 8 位)
        """
        run_id = str(uuid.uuid4())[:8]

        if self._relational is not None:
            self._relational.save_pipeline_run(
                run_id=run_id,
                experiment_id=experiment_id,
                temporal_method=temporal_method,
                feature_space_method=feature_space_method,
                config_snapshot=config_snapshot,
                n_modalities_processed=n_modalities_processed,
                steps_executed=steps_executed,
            )

        logg.info(f"对齐运行已记录: {run_id} (experiment={experiment_id})")
        return run_id

    def mark_run_completed(self, run_id: str) -> None:
        """标记流水线运行完成"""
        if self._relational is not None:
            self._relational.mark_run_completed(run_id)

    def mark_run_failed(self, run_id: str, error_message: str = "") -> None:
        """标记流水线运行失败"""
        if self._relational is not None:
            self._relational.mark_run_failed(run_id, error_message)

    # ------------------------------------------------------------------
    # High-level workflow methods — 质量指标
    # ------------------------------------------------------------------

    def save_alignment_metrics(
        self,
        run_id: str,
        metrics: dict[str, dict[str, float]],
    ) -> None:
        """保存三维度对齐质量指标

        Args:
            run_id: 运行 ID
            metrics: {dimension: {metric_name: value}}
                例如 {"time_consistency": {"dtw_distance_reduction": 0.3, ...}, ...}
        """
        if self._relational is not None:
            self._relational.save_metrics_batch(run_id, metrics)
            logg.info(f"质量指标已保存: {run_id} ({sum(len(v) for v in metrics.values())} 个指标)")

    def save_single_metric(
        self,
        run_id: str,
        dimension: str,
        metric_name: str,
        metric_value: float,
        metric_detail: dict[str, Any] | None = None,
    ) -> None:
        """保存单条质量指标"""
        if self._relational is not None:
            self._relational.save_metric(run_id, dimension, metric_name, metric_value, metric_detail)

    # ------------------------------------------------------------------
    # High-level workflow methods — 对象存储
    # ------------------------------------------------------------------

    def put_aligned_mudata(self, experiment_id: str, path: str | Path) -> str:
        """上传对齐后 MuData 到对象存储"""
        if self._minio is None:
            raise RuntimeError("存储管理器未连接，请先调用 connect()")
        return self._minio.put_aligned_mudata(experiment_id, path)

    def put_feature_matrix(self, experiment_id: str, modality: str, path: str | Path) -> str:
        """上传特征矩阵"""
        if self._minio is None:
            raise RuntimeError("存储管理器未连接")
        return self._minio.put_feature_matrix(experiment_id, modality, path)

    def put_integrated_matrix(self, experiment_id: str, path: str | Path) -> str:
        """上传跨模态集成矩阵"""
        if self._minio is None:
            raise RuntimeError("存储管理器未连接")
        return self._minio.put_integrated_matrix(experiment_id, path)

    def put_alignment_report(self, experiment_id: str, path: str | Path, fmt: str = "json") -> str:
        """上传对齐报告"""
        if self._minio is None:
            raise RuntimeError("存储管理器未连接")
        return self._minio.put_report(experiment_id, path, fmt)

    # ------------------------------------------------------------------
    # High-level workflow methods — 知识图谱
    # ------------------------------------------------------------------

    def build_alignment_knowledge_graph(
        self,
        mdata: MuData,
        experiment_id: str,
        temporal_method: str | None = None,
        feature_space_method: str | None = None,
    ) -> None:
        """从 MuData 构建对齐知识图谱

        创建 Experiment/Modality/TimePoint/Batch 节点及对齐关系。
        """
        if self._graph is not None:
            self._graph.build_alignment_knowledge_graph(
                mdata,
                experiment_id=experiment_id,
                temporal_method=temporal_method,
                feature_space_method=feature_space_method,
            )
            logg.info(f"对齐知识图谱已构建: {experiment_id}")

    def build_trajectory_graph(self, mdata: MuData, experiment_id: str) -> None:
        """从 MuData 构建轨迹图"""
        if self._graph is not None:
            self._graph.build_trajectory_graph(mdata, experiment_id)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def export_experiment(self, experiment_id: str) -> dict[str, Any]:
        """导出实验的完整数据快照（关系型 + 对象存储 + 图）"""
        result: dict[str, Any] = {
            "experiment_id": experiment_id,
        }

        # 从关系型数据库导出
        if self._relational is not None:
            result["relational"] = self._relational.export_experiment(experiment_id)

        # 列出对象存储中的文件
        if self._minio is not None:
            result["objects"] = self._minio.list(f"alignments/{experiment_id}/")

        # 图数据库中的相关节点
        if self._graph is not None:
            result["graph_modalities"] = self._graph.query_nodes(
                "Modality", {"experiment_id": experiment_id}
            )
            result["graph_timepoints"] = self._graph.query_nodes(
                "TimePoint", {"experiment_id": experiment_id}
            )

        return result

    def delete_experiment(self, experiment_id: str) -> None:
        """删除实验的所有数据（级联删除）"""
        if self._relational is not None:
            self._relational.delete_experiment(experiment_id)
        if self._minio is not None:
            self._minio.delete_experiment(experiment_id)
        logg.info(f"实验已完全删除: {experiment_id}")

    def health_check(self) -> dict[str, bool]:
        """检查所有存储后端的健康状态"""
        return {
            "minio": self._minio.is_healthy() if self._minio else False,
            "relational": self._relational.is_healthy() if self._relational else False,
            "graph": self._graph.is_healthy() if self._graph else False,
        }

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """获取实验的完整摘要（实验信息 + 模态汇总 + 最新运行 + 最新指标）"""
        summary: dict[str, Any] = {"experiment_id": experiment_id}

        if self._relational is not None:
            summary["experiment"] = self._relational.get_experiment(experiment_id)
            summary["modality_summary"] = self._relational.get_modality_summary(experiment_id)
            runs = self._relational.query_pipeline_runs(experiment_id=experiment_id)
            if runs:
                summary["latest_run"] = runs[0]
                summary["latest_metrics"] = self._relational.get_metrics(runs[0]["run_id"])
            summary["feature_matrices"] = self._relational.list_feature_matrices(experiment_id)

        if self._minio is not None:
            summary["stored_objects"] = self._minio.list(f"alignments/{experiment_id}/")

        return summary
