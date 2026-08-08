"""混合存储架构模块

节点分离的混合存储层:
    - MinIO / S3 兼容对象存储: 大型对齐 MuData / 特征矩阵 CSV / 报告文件
    - 关系型数据库 (SQLite / DM8): 实验元数据、模态详情、流水线运行、质量指标
    - 图数据库 (Neo4j): 模态对齐关系、跨模态关联图、时间轨迹、批次效应

用法:
    from storage import StorageManager

    store = StorageManager.from_config("config/default.yaml")

    with store:
        # 记录对齐实验
        store.save_experiment("E001", temporal_method="dtw", feature_space_method="cca")

        # 上传对齐后 MuData
        store.put_aligned_mudata("E001", "data/aligned/aligned.h5mu")

        # 记录流水线运行
        run_id = store.record_alignment_run("E001", temporal_method="dtw")

        # 保存评估指标
        store.save_alignment_metrics(run_id, {
            "time_consistency": {"dtw_distance_reduction": 0.35},
        })

        # 构建知识图谱
        store.build_alignment_knowledge_graph(mdata, "E001")

注意: 当前存储模块已编写完成但未接入流水线。
      未来可通过 pipeline._pipeline 中的 use_storage 参数启用。
"""

from __future__ import annotations

from ._base import BaseStorageClient
from ._minio import MinIOClient
from ._relational import RelationalDBClient
from ._graph import GraphDBClient
from ._manager import StorageManager

__all__ = [
    "BaseStorageClient",
    "MinIOClient",
    "RelationalDBClient",
    "GraphDBClient",
    "StorageManager",
]
