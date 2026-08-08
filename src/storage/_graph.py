"""
图数据库客户端 (Neo4j)

存储对齐关系型数据:
    - 模态间对齐关系 (modality_A) -[:ALIGNED_TO {method: "dtw"}]-> (modality_B)
    - 时间轨迹关系 (timepoint_t1) -[:PRECEDES]-> (timepoint_t2)
    - 跨模态关联图 (transcriptomic ↔ proteomic ↔ metabolomic)
    - 批次效应知识图谱 (batch → instrument → date → operator)
    - 特征空间校正关系 (pre_alignment) -[:CORRECTED_BY {method: "mnn"}]-> (post_alignment)

未安装 Neo4j 驱动时使用 JSON-LD 文件作为 fallback。
与 Module 1 的 storage._graph 保持接口兼容。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _logging as logg
from ._base import BaseStorageClient

if TYPE_CHECKING:
    pass


class GraphDBClient(BaseStorageClient):
    """Neo4j 图数据库客户端

    存储节点与关系，支持:
        - Cypher 查询
        - 批量节点/关系创建
        - 路径查询 (shortest path, BFS/DFS)
        - 对齐知识图谱构建

    未安装 neo4j 驱动时使用 JSON-LD 文件 fallback。

    节点类型 (Node Types):
        - Experiment:   对齐实验节点 (experiment_id, temporal_method, feature_space_method, ...)
        - Modality:     模态节点 (modality_name, n_obs, n_features, time_unit, ...)
        - TimePoint:    时间点节点 (time_value, modality, n_samples)
        - Trajectory:   轨迹节点 (trajectory_id, n_timepoints, method)
        - Batch:        批次节点 (batch_id, ...)
        - Feature:      特征节点 (feature_id, feature_name, modality, ...)

    关系类型 (Relationship Types):
        - ALIGNED_TO:             模态间对齐关系 (Modality → Modality)
        - CROSS_MODALITY_CORRELATED: 跨模态特征关联 (Feature → Feature)
        - PRECEDES:               时间先后关系 (TimePoint → TimePoint)
        - BELONGS_TO_EXPERIMENT:  归属实验 (Modality|Trajectory → Experiment)
        - SAME_TRAJECTORY:        同轨迹 (Modality → Trajectory)
        - CORRECTED_BY:           批次校正 (Modality → Modality)
        - HAS_TIMEPOINT:          模态含时间点 (Modality → TimePoint)
        - BELONGS_TO_BATCH:       归属批次 (Modality → Batch)

    用法:
        graph = GraphDBClient(config={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
            "database": "omics",
        })
        with graph:
            graph.create_experiment_node("E001", temporal_method="dtw", feature_space_method="cca")
            graph.create_modality_node("scrna", experiment_id="E001", n_obs=5000)
            graph.link_modalities("scrna", "proteomics", relation="ALIGNED_TO", method="cca")
            path = graph.find_path("scrna", "metabolomics", max_depth=5)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._driver: Any = None
        self._database: str = self.config.get("database", "omics")
        self._local_fallback: Path | None = None
        self._graph_data: dict[str, Any] = {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立 Neo4j 连接，失败则使用本地 JSON 文件 fallback"""
        try:
            from neo4j import GraphDatabase

            uri = self.config.get("uri", "bolt://localhost:7687")
            user = self.config.get("user", "neo4j")
            password = self.config.get("password", "neo4j")

            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # 验证连接
            with self._driver.session(database=self._database) as session:
                session.run("RETURN 1")
            self._connected = True
            logg.info(f"Neo4j 已连接 (uri={uri}, db={self._database})")

        except ImportError:
            logg.warning("neo4j 驱动未安装，使用 JSON-LD 文件 fallback")
            self._setup_local_fallback()
        except Exception as exc:
            logg.warning(f"Neo4j 连接失败 ({exc})，使用 JSON-LD 文件 fallback")
            self._setup_local_fallback()

    def _setup_local_fallback(self) -> None:
        """设置本地 JSON-LD 文件 fallback"""
        fallback_dir = self.config.get("fallback_dir", "data/storage/graph")
        self._local_fallback = Path(fallback_dir)
        self._local_fallback.mkdir(parents=True, exist_ok=True)
        # 加载已有图数据
        graph_file = self._local_fallback / "graph.json"
        if graph_file.exists():
            try:
                self._graph_data = json.loads(graph_file.read_text("utf-8"))
            except json.JSONDecodeError:
                self._graph_data = {"nodes": [], "edges": []}
        self._connected = True
        logg.info(f"本地图存储 fallback: {self._local_fallback}")

    def disconnect(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
        # 持久化本地 fallback 数据
        self._flush_local()
        self._connected = False

    def _flush_local(self) -> None:
        """将内存中的图数据写入本地 JSON 文件"""
        if self._local_fallback is not None and self._graph_data:
            graph_file = self._local_fallback / "graph.json"
            graph_file.write_text(
                json.dumps(self._graph_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def is_healthy(self) -> bool:
        if not self._connected:
            return False
        if self._driver:
            try:
                with self._driver.session(database=self._database) as session:
                    session.run("RETURN 1")
                return True
            except Exception:
                return False
        return self._local_fallback is not None and self._local_fallback.is_dir()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None = None) -> str:
        return key

    def get(self, key: str) -> bytes | None:
        return None

    def list(self, prefix: str = "") -> list[str]:
        labels: list[str] = []
        for node in self._graph_data["nodes"]:
            for label in node.get("labels", []):
                if label.startswith(prefix) and label not in labels:
                    labels.append(label)
        return labels

    def delete(self, key: str) -> bool:
        if self._driver:
            with self._driver.session(database=self._database) as session:
                session.run(f"MATCH (n:{key}) DETACH DELETE n")
        self._graph_data["nodes"] = [n for n in self._graph_data["nodes"] if n.get("id") != key]
        self._graph_data["edges"] = [e for e in self._graph_data["edges"] if e["from"] != key and e["to"] != key]
        return True

    def exists(self, key: str) -> bool:
        return any(n.get("id") == key for n in self._graph_data["nodes"])

    # ------------------------------------------------------------------
    # Node operations — 对齐领域节点
    # ------------------------------------------------------------------

    def create_experiment_node(
        self,
        experiment_id: str,
        temporal_method: str | None = None,
        feature_space_method: str | None = None,
        n_modalities: int | None = None,
        n_total_obs: int | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """创建对齐实验节点

        Args:
            experiment_id: 实验唯一 ID
            temporal_method: 时间对齐方法 (dtw, interpolation, pseudotime, lag)
            feature_space_method: 特征空间校正方法 (mnn, cca, optimal_transport, manifold)
            n_modalities: 模态数
            n_total_obs: 总观测数
            properties: 额外属性

        Returns:
            experiment_id
        """
        props: dict[str, Any] = {
            "experiment_id": experiment_id,
            "temporal_method": temporal_method or "auto",
            "feature_space_method": feature_space_method or "auto",
        }
        if n_modalities is not None:
            props["n_modalities"] = n_modalities
        if n_total_obs is not None:
            props["n_total_obs"] = n_total_obs
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                """
                MERGE (e:Experiment {experiment_id: $experiment_id})
                SET e = $props
                RETURN e
                """,
                {"experiment_id": experiment_id, "props": props},
            )
        else:
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == experiment_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": experiment_id,
                    "type": "Experiment",
                    "labels": ["Experiment"],
                    "properties": props,
                })

        logg.info(f"图节点已创建: Experiment({experiment_id})")
        return experiment_id

    def create_modality_node(
        self,
        modality_name: str,
        experiment_id: str | None = None,
        n_obs: int | None = None,
        n_features: int | None = None,
        n_time_points: int | None = None,
        time_unit: str | None = None,
        n_batches: int | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """创建模态节点

        Args:
            modality_name: 模态名称 (scrna, proteomics, metabolomics, ...)
            experiment_id: 所属实验 ID
            n_obs: 观测数
            n_features: 特征维度
            n_time_points: 时间点数
            time_unit: 时间单位 (hour, day)
            n_batches: 批次数
            properties: 额外属性

        Returns:
            node_id (experiment_id/modality_name 或 modality_name)
        """
        node_id = f"{experiment_id}/{modality_name}" if experiment_id else modality_name

        props: dict[str, Any] = {
            "modality_name": modality_name,
            "experiment_id": experiment_id or "unknown",
        }
        if n_obs is not None:
            props["n_obs"] = n_obs
        if n_features is not None:
            props["n_features"] = n_features
        if n_time_points is not None:
            props["n_time_points"] = n_time_points
        if time_unit is not None:
            props["time_unit"] = time_unit
        if n_batches is not None:
            props["n_batches"] = n_batches
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                """
                MERGE (m:Modality {modality_name: $modality_name, experiment_id: $experiment_id})
                SET m = $props
                RETURN m
                """,
                {"modality_name": modality_name, "experiment_id": experiment_id or "unknown", "props": props},
            )
        else:
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == node_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": node_id,
                    "type": "Modality",
                    "labels": ["Modality"],
                    "properties": props,
                })

        logg.info(f"图节点已创建: Modality({node_id})")
        return node_id

    def create_timepoint_node(
        self,
        time_value: float,
        modality_name: str,
        experiment_id: str | None = None,
        n_samples: int | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """创建时间点节点

        Args:
            time_value: 对齐后的时间值
            modality_name: 所属模态
            experiment_id: 所属实验 ID
            n_samples: 该时间点的样本数
            properties: 额外属性

        Returns:
            node_id
        """
        node_id = f"{experiment_id}/{modality_name}/t{time_value}"

        props: dict[str, Any] = {
            "time_value": time_value,
            "modality_name": modality_name,
            "experiment_id": experiment_id or "unknown",
        }
        if n_samples is not None:
            props["n_samples"] = n_samples
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                """
                MERGE (tp:TimePoint {time_value: $time_value, modality_name: $modality_name, experiment_id: $experiment_id})
                SET tp = $props
                RETURN tp
                """,
                {"time_value": time_value, "modality_name": modality_name, "experiment_id": experiment_id or "unknown", "props": props},
            )
        else:
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == node_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": node_id,
                    "type": "TimePoint",
                    "labels": ["TimePoint"],
                    "properties": props,
                })

        return node_id

    def create_trajectory_node(
        self,
        trajectory_id: str,
        experiment_id: str | None = None,
        n_timepoints: int | None = None,
        method: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """创建轨迹节点（伪时间/过程轨迹）

        Args:
            trajectory_id: 轨迹 ID
            experiment_id: 所属实验
            n_timepoints: 时间点数量
            method: 轨迹推断方法 (pseudotime, dtw_path, lag_chain)
            properties: 额外属性

        Returns:
            trajectory_id
        """
        props: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "experiment_id": experiment_id or "unknown",
        }
        if n_timepoints is not None:
            props["n_timepoints"] = n_timepoints
        if method is not None:
            props["method"] = method
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                """
                MERGE (t:Trajectory {trajectory_id: $trajectory_id})
                SET t = $props
                RETURN t
                """,
                {"trajectory_id": trajectory_id, "props": props},
            )
        else:
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == trajectory_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": trajectory_id,
                    "type": "Trajectory",
                    "labels": ["Trajectory"],
                    "properties": props,
                })

        logg.info(f"图节点已创建: Trajectory({trajectory_id})")
        return trajectory_id

    def create_batch_node(self, batch_id: str, properties: dict[str, Any] | None = None) -> str:
        """创建批次节点"""
        props = {"batch_id": batch_id}
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                "MERGE (b:Batch {batch_id: $batch_id}) SET b = $props",
                {"batch_id": batch_id, "props": props},
            )
        else:
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == batch_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": batch_id,
                    "type": "Batch",
                    "labels": ["Batch"],
                    "properties": props,
                })

        return batch_id

    def create_feature_node(
        self,
        feature_id: str,
        feature_name: str | None = None,
        modality_name: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """创建特征节点（基因/蛋白/代谢物）

        Args:
            feature_id: 特征唯一 ID (ENSG..., protein_id, metabolite_id)
            feature_name: 特征名称/符号
            modality_name: 所属模态
            properties: 额外属性

        Returns:
            feature_id
        """
        props: dict[str, Any] = {
            "feature_id": feature_id,
            "feature_name": feature_name or feature_id,
        }
        if modality_name:
            props["modality_name"] = modality_name
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                """
                MERGE (f:Feature {feature_id: $feature_id})
                SET f = $props
                RETURN f
                """,
                {"feature_id": feature_id, "props": props},
            )
        else:
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == feature_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": feature_id,
                    "type": "Feature",
                    "labels": ["Feature"],
                    "properties": props,
                })

        logg.info(f"图节点已创建: Feature({feature_id})")
        return feature_id

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def link_modalities(
        self,
        source_modality: str,
        target_modality: str,
        relation: str = "ALIGNED_TO",
        method: str | None = None,
        score: float | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """创建模态间对齐关系

        Args:
            source_modality: 源模态名称
            target_modality: 目标模态名称
            relation: 关系类型 (ALIGNED_TO, CROSS_MODALITY_CORRELATED, CORRECTED_BY)
            method: 对齐方法
            score: 对齐质量评分
            properties: 额外属性
        """
        props = properties or {}
        if method is not None:
            props["method"] = method
        if score is not None:
            props["score"] = score

        if self._driver:
            rel_upper = relation.upper()
            self._run_cypher(
                f"""
                MATCH (a:Modality {{modality_name: $source}})
                MATCH (b:Modality {{modality_name: $target}})
                MERGE (a)-[r:{rel_upper}]->(b)
                SET r = $props
                """,
                {"source": source_modality, "target": target_modality, "props": props},
            )
        else:
            self._graph_data["edges"].append({
                "from": source_modality,
                "to": target_modality,
                "type": relation,
                "properties": props,
            })
            self._flush_local()

        logg.info(f"图关系已创建: ({source_modality})-[:{relation}]->({target_modality})")

    def link_timepoints(
        self,
        source_tp_id: str,
        target_tp_id: str,
        relation: str = "PRECEDES",
        time_delta: float | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """创建时间点先后关系

        Args:
            source_tp_id: 前驱时间点节点 ID
            target_tp_id: 后继时间点节点 ID
            relation: 关系类型 (PRECEDES)
            time_delta: 时间差
            properties: 额外属性
        """
        props = properties or {}
        if time_delta is not None:
            props["time_delta"] = time_delta

        if self._driver:
            rel_upper = relation.upper()
            self._run_cypher(
                f"""
                MATCH (a:TimePoint)
                MATCH (b:TimePoint)
                WHERE a.time_value = $source_val AND b.time_value = $target_val
                MERGE (a)-[r:{rel_upper}]->(b)
                SET r = $props
                """,
                {"source_val": source_tp_id, "target_val": target_tp_id, "props": props},
            )
        else:
            self._graph_data["edges"].append({
                "from": source_tp_id,
                "to": target_tp_id,
                "type": relation,
                "properties": props,
            })
            self._flush_local()

        logg.info(f"图关系已创建: ({source_tp_id})-[:{relation}]->({target_tp_id})")

    def _create_relationship(
        self,
        source_id: str,
        target_id: str,
        source_label: str,
        target_label: str,
        source_id_field: str,
        target_id_field: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """通用关系创建方法（内部使用）

        支持任意节点类型间的关系，同时兼容 Neo4j 和 JSON-LD fallback。

        Args:
            source_id: 源节点 ID
            target_id: 目标节点 ID
            source_label: 源节点标签（如 Modality, Experiment）
            target_label: 目标节点标签（如 Modality, Trajectory）
            source_id_field: 源节点的 ID 属性名
            target_id_field: 目标节点的 ID 属性名
            relation: 关系类型
            properties: 关系属性
        """
        props = properties or {}
        rel_upper = relation.upper()

        if self._driver:
            self._run_cypher(
                f"""
                MATCH (a:{source_label} {{{source_id_field}: $source_id}})
                MATCH (b:{target_label} {{{target_id_field}: $target_id}})
                MERGE (a)-[r:{rel_upper}]->(b)
                SET r = $props
                """,
                {"source_id": source_id, "target_id": target_id, "props": props},
            )
        else:
            self._graph_data["edges"].append({
                "from": source_id,
                "to": target_id,
                "type": relation,
                "properties": props,
            })
            self._flush_local()

        logg.info(f"图关系已创建: ({source_id}:{source_label})-[:{relation}]->({target_id}:{target_label})")

    def link_modality_to_experiment(
        self,
        modality_name: str,
        experiment_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """将模态关联到实验 (Modality → Experiment)"""
        self._create_relationship(
            source_id=modality_name,
            target_id=experiment_id,
            source_label="Modality",
            target_label="Experiment",
            source_id_field="modality_name",
            target_id_field="experiment_id",
            relation="BELONGS_TO_EXPERIMENT",
            properties=properties,
        )

    def link_modality_to_trajectory(
        self,
        modality_name: str,
        trajectory_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """将模态关联到轨迹 (Modality → Trajectory)"""
        self._create_relationship(
            source_id=modality_name,
            target_id=trajectory_id,
            source_label="Modality",
            target_label="Trajectory",
            source_id_field="modality_name",
            target_id_field="trajectory_id",
            relation="SAME_TRAJECTORY",
            properties=properties,
        )

    def link_modality_to_batch(
        self,
        modality_name: str,
        batch_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """将模态关联到批次 (Modality → Batch)"""
        self._create_relationship(
            source_id=modality_name,
            target_id=batch_id,
            source_label="Modality",
            target_label="Batch",
            source_id_field="modality_name",
            target_id_field="batch_id",
            relation="BELONGS_TO_BATCH",
            properties=properties,
        )

    def link_modality_timepoints(
        self,
        modality_name: str,
        timepoint_ids: list[str],
        properties: dict[str, Any] | None = None,
    ) -> None:
        """批量创建模态到时间点的关系 (Modality → TimePoint)"""
        for tp_id in timepoint_ids:
            self._create_relationship(
                source_id=modality_name,
                target_id=tp_id,
                source_label="Modality",
                target_label="TimePoint",
                source_id_field="modality_name",
                target_id_field="time_value",
                relation="HAS_TIMEPOINT",
                properties=properties,
            )

    def link_cross_modality_features(
        self,
        source_feature_id: str,
        target_feature_id: str,
        correlation: float | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """创建跨模态特征关联 (Feature → Feature)

        Args:
            source_feature_id: 源特征 ID
            target_feature_id: 目标特征 ID（另一模态）
            correlation: 跨模态相关系数
            properties: 额外属性
        """
        props = properties or {}
        if correlation is not None:
            props["correlation"] = correlation

        self._create_relationship(
            source_id=source_feature_id,
            target_id=target_feature_id,
            source_label="Feature",
            target_label="Feature",
            source_id_field="feature_id",
            target_id_field="feature_id",
            relation="CROSS_MODALITY_CORRELATED",
            properties=props,
        )

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        node_label: str = "Modality",
        id_field: str = "modality_name",
    ) -> list[dict[str, Any]]:
        """查找两个节点之间的最短路径

        Returns:
            路径列表，每个元素为 {nodes: [...], relationships: [...]}
        """
        if self._driver:
            result = self._run_cypher(
                f"""
                MATCH path = shortestPath(
                    (a:{node_label} {{{id_field}: $source_id}})-[*1..$max_depth]-(b:{node_label} {{{id_field}: $target_id}})
                )
                RETURN path LIMIT 1
                """,
                {"source_id": source_id, "target_id": target_id, "max_depth": max_depth},
            )
            return [dict(r) for r in result]
        else:
            return self._local_bfs(source_id, target_id, max_depth)

    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        depth: int = 1,
        node_label: str = "Modality",
        id_field: str = "modality_name",
    ) -> list[dict[str, Any]]:
        """获取节点的邻居节点

        Args:
            node_id: 节点 ID
            relation: 限定关系类型（None 表示所有类型）
            depth: 邻居深度（1 = 直接邻居）
            node_label: 节点标签
            id_field: 节点的 ID 属性名
        """
        if self._driver:
            rel_clause = f":{relation.upper()}" if relation else ""
            result = self._run_cypher(
                f"""
                MATCH (n:{node_label} {{{id_field}: $node_id}})-[r{rel_clause}*1..$depth]-(neighbor)
                RETURN DISTINCT neighbor, r
                """,
                {"node_id": node_id, "depth": depth},
            )
            return [dict(r) for r in result]
        else:
            neighbors: list[dict[str, Any]] = []
            for edge in self._graph_data["edges"]:
                if relation and edge["type"] != relation:
                    continue
                if edge["from"] == node_id:
                    target = next((n for n in self._graph_data["nodes"] if n["id"] == edge["to"]), None)
                    if target:
                        neighbors.append({"node": target, "relationship": edge})
                elif edge["to"] == node_id:
                    source = next((n for n in self._graph_data["nodes"] if n["id"] == edge["from"]), None)
                    if source:
                        neighbors.append({"node": source, "relationship": edge})
            return neighbors[:depth * 10]

    def get_aligned_modalities(
        self,
        modality_name: str,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """获取与给定模态对齐的其他模态"""
        neighbors = self.get_neighbors(modality_name, relation="ALIGNED_TO")
        if min_score > 0:
            return [
                n for n in neighbors
                if n.get("relationship", {}).get("properties", {}).get("score", 0) >= min_score
            ]
        return neighbors

    def query_nodes(
        self,
        label: str,
        properties: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按标签和属性查询节点"""
        if self._driver:
            where_clauses = []
            params: dict[str, Any] = {"limit": limit}
            if properties:
                for i, (k, v) in enumerate(properties.items()):
                    param_key = f"prop_{i}"
                    where_clauses.append(f"n.{k} = ${param_key}")
                    params[param_key] = v

            where_str = " AND ".join(where_clauses)
            if where_str:
                where_str = "WHERE " + where_str

            result = self._run_cypher(
                f"MATCH (n:{label}) {where_str} RETURN n LIMIT $limit",
                params,
            )
            return [dict(r) for r in result]
        else:
            matches = []
            for node in self._graph_data["nodes"]:
                if label not in node.get("labels", []):
                    continue
                if properties:
                    node_props = node.get("properties", {})
                    if all(node_props.get(k) == v for k, v in properties.items()):
                        matches.append(node)
                else:
                    matches.append(node)
            return matches[:limit]

    # ------------------------------------------------------------------
    # Knowledge graph construction — 对齐知识图谱
    # ------------------------------------------------------------------

    def build_alignment_knowledge_graph(
        self,
        mdata: Any,
        experiment_id: str,
        temporal_method: str | None = None,
        feature_space_method: str | None = None,
    ) -> None:
        """从 MuData 构建对齐知识图谱

        节点: Experiment, Modality, TimePoint, Batch
        关系: BELONGS_TO_EXPERIMENT, HAS_TIMEPOINT, PRECEDES, BELONGS_TO_BATCH,
              ALIGNED_TO (跨模态)

        Args:
            mdata: 对齐后的 MuData 对象
            experiment_id: 实验 ID
            temporal_method: 时间对齐方法
            feature_space_method: 特征空间校正方法
        """
        # 1. 创建实验节点
        modalities = list(mdata.mod.keys())
        n_total_obs = sum(adata.n_obs for adata in mdata.mod.values())
        self.create_experiment_node(
            experiment_id=experiment_id,
            temporal_method=temporal_method,
            feature_space_method=feature_space_method,
            n_modalities=len(modalities),
            n_total_obs=n_total_obs,
        )

        # 2. 为每个模态创建节点和时间点
        time_key = "aligned_time"
        batch_key = "batch"

        for mod_name, adata in mdata.mod.items():
            n_features = adata.n_vars
            if "X_feature_aligned" in adata.obsm:
                n_features = adata.obsm["X_feature_aligned"].shape[1]

            n_time_points = 0
            time_unit = None
            if time_key in adata.obs.columns:
                n_time_points = adata.obs[time_key].nunique()
            if "time_unit" in adata.obs.columns:
                time_unit = str(adata.obs["time_unit"].iloc[0]) if adata.n_obs > 0 else None

            n_batches = 0
            if batch_key in adata.obs.columns:
                n_batches = adata.obs[batch_key].nunique()

            self.create_modality_node(
                modality_name=mod_name,
                experiment_id=experiment_id,
                n_obs=adata.n_obs,
                n_features=n_features,
                n_time_points=n_time_points,
                time_unit=time_unit,
                n_batches=n_batches,
            )

            # 关联模态到实验
            self.link_modality_to_experiment(mod_name, experiment_id)

            # 创建时间点节点
            if time_key in adata.obs.columns:
                time_values = sorted(adata.obs[time_key].unique())
                tp_ids = []
                for tv in time_values:
                    n_samples = int((adata.obs[time_key] == tv).sum())
                    tp_id = self.create_timepoint_node(
                        time_value=float(tv),
                        modality_name=mod_name,
                        experiment_id=experiment_id,
                        n_samples=n_samples,
                    )
                    tp_ids.append(tp_id)

                # 创建时间先后关系
                for i in range(len(tp_ids) - 1):
                    self.link_timepoints(
                        tp_ids[i], tp_ids[i + 1],
                        time_delta=float(time_values[i + 1]) - float(time_values[i]),
                    )

                # 关联模态到时间点
                self.link_modality_timepoints(mod_name, tp_ids)

            # 创建批次节点和关系
            if batch_key in adata.obs.columns:
                batches = adata.obs[batch_key].unique()
                for batch in batches:
                    batch_id = f"{experiment_id}/{mod_name}/batch_{batch}"
                    self.create_batch_node(batch_id, {
                        "batch_label": str(batch),
                        "modality": mod_name,
                        "n_samples": int((adata.obs[batch_key] == batch).sum()),
                    })
                    self.link_modality_to_batch(mod_name, batch_id)

        # 3. 创建跨模态对齐关系
        for i, mod_a in enumerate(modalities):
            for mod_b in modalities[i + 1:]:
                self.link_modalities(
                    mod_a, mod_b,
                    relation="ALIGNED_TO",
                    method=f"{temporal_method}+{feature_space_method}",
                )

        logg.info(f"对齐知识图谱已构建: {len(modalities)} 个模态, experiment={experiment_id}")

    def build_trajectory_graph(
        self,
        mdata: Any,
        experiment_id: str,
    ) -> None:
        """从伪时间/DTW warping path 构建轨迹图

        为每个模态创建 Trajectory 节点，关联 TimePoint 节点形成轨迹路径。
        """
        time_key = "aligned_time"

        for mod_name, adata in mdata.mod.items():
            if time_key not in adata.obs.columns:
                continue

            traj_id = f"{experiment_id}/{mod_name}/trajectory"
            time_values = sorted(adata.obs[time_key].unique())

            self.create_trajectory_node(
                trajectory_id=traj_id,
                experiment_id=experiment_id,
                n_timepoints=len(time_values),
                method="pseudotime",
            )

            # 关联模态到轨迹
            self.link_modality_to_trajectory(mod_name, traj_id)

        logg.info(f"轨迹图已构建: experiment={experiment_id}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_cypher(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """执行 Cypher 查询"""
        if self._driver is None:
            logg.warning("图数据库未连接")
            return []
        with self._driver.session(database=self._database) as session:
            return list(session.run(query, params or {}))

    def _local_bfs(self, source_id: str, target_id: str, max_depth: int) -> list[dict[str, Any]]:
        """本地 BFS 路径搜索（fallback 模式）"""
        from collections import deque

        # 构建邻接表
        adjacency: dict[str, list[tuple[str, dict]]] = {}
        for edge in self._graph_data["edges"]:
            frm, to = edge["from"], edge["to"]
            adjacency.setdefault(frm, []).append((to, edge))
            adjacency.setdefault(to, []).append((frm, edge))

        if source_id not in adjacency:
            return []

        queue: deque[tuple[str, list[str], list[dict]]] = deque()
        queue.append((source_id, [source_id], []))
        visited: set[str] = {source_id}

        while queue:
            current, path_nodes, path_edges = queue.popleft()
            if len(path_nodes) - 1 > max_depth:
                continue

            if current == target_id:
                return [{"nodes": path_nodes, "relationships": path_edges}]

            for neighbor, edge in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path_nodes + [neighbor],
                        path_edges + [edge],
                    ))

        return []
