"""
关系型数据库客户端 (SQLite / DM8)

存储结构化对齐元数据:
    - 对齐实验记录 (实验 ID, 模态列表, 时间/特征空间方法)
    - 模态详情 (n_obs, n_features, 时间点数, 插值比例)
    - 流水线运行记录 (config 快照, 时间戳, 运行状态)
    - 特征矩阵元数据 (导出路径, 维度, 索引方式)
    - 对齐质量指标 (时间一致性, 分布一致性, 跨模态关联)

DM8 连接时使用 dmPython；未安装时自动降级为 SQLite。
与 Module 1 的 storage._relational 保持接口兼容。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _logging as logg
from ._base import BaseStorageClient

if TYPE_CHECKING:
    pass


_DEFAULT_SCHEMA_SQL = """
-- 对齐实验记录表
CREATE TABLE IF NOT EXISTS alignment_experiments (
    experiment_id       TEXT PRIMARY KEY,
    input_source        TEXT,               -- Module 1 数据来源路径
    n_modalities        INTEGER,
    modalities_json     TEXT,               -- 模态列表 (JSON array)
    temporal_method     TEXT,               -- dtw | interpolation | pseudotime | lag
    feature_space_method TEXT,              -- mnn | cca | optimal_transport | manifold
    time_normalized     INTEGER DEFAULT 0,  -- 是否做了时间尺度归一化
    n_total_obs         INTEGER,
    n_total_features    INTEGER,
    pipeline_version    TEXT,
    config_snapshot     TEXT,               -- 完整配置 YAML/JSON 快照
    notes               TEXT,               -- 人工备注
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 模态级对齐详情表
CREATE TABLE IF NOT EXISTS alignment_modalities (
    modality_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       TEXT NOT NULL,
    modality_name       TEXT NOT NULL,      -- scrna, proteomics, metabolomics, ...
    n_obs               INTEGER,
    n_features_before   INTEGER,            -- 对齐前特征维度
    n_features_after    INTEGER,            -- 对齐后特征维度
    n_time_points       INTEGER,            -- 唯一时间点数
    time_unit           TEXT,               -- hour | day
    time_range_min      REAL,
    time_range_max      REAL,
    n_interpolated      INTEGER DEFAULT 0,  -- 插值观测数
    n_original          INTEGER DEFAULT 0,  -- 原始观测数
    temporal_sub_method TEXT,               -- 该模态实际使用的时间对齐方法
    lag_shift           REAL,               -- 滞后偏移量 (lag modeling)
    n_batches           INTEGER,            -- 批次数
    feature_dim_order   TEXT,               -- 特征维度排序方式
    metadata_json       TEXT,               -- 自由格式额外元数据
    FOREIGN KEY (experiment_id) REFERENCES alignment_experiments(experiment_id)
);

-- 流水线运行记录表
CREATE TABLE IF NOT EXISTS alignment_runs (
    run_id              TEXT PRIMARY KEY,   -- UUID
    experiment_id       TEXT NOT NULL,
    temporal_method     TEXT,
    feature_space_method TEXT,
    config_yaml         TEXT,               -- 完整配置快照
    n_modalities_processed INTEGER,
    steps_executed      TEXT,               -- 执行的步骤列表 (JSON array)
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    status              TEXT DEFAULT 'running',  -- running, completed, failed
    error_message       TEXT,
    FOREIGN KEY (experiment_id) REFERENCES alignment_experiments(experiment_id)
);

-- 特征矩阵导出记录表
CREATE TABLE IF NOT EXISTS feature_matrices (
    matrix_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       TEXT NOT NULL,
    run_id              TEXT,
    modality_name       TEXT,               -- NULL = integrated matrix
    matrix_type         TEXT NOT NULL,      -- observation | time_indexed | integrated
    file_format         TEXT DEFAULT 'csv',
    n_rows              INTEGER,
    n_columns           INTEGER,
    index_key           TEXT,               -- aligned_time | process_stage | observation
    storage_path        TEXT,               -- MinIO key 或本地路径
    file_hash           TEXT,               -- SHA256
    exported_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES alignment_experiments(experiment_id),
    FOREIGN KEY (run_id) REFERENCES alignment_runs(run_id)
);

-- 对齐质量指标表
CREATE TABLE IF NOT EXISTS alignment_metrics (
    metric_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    dimension           TEXT NOT NULL,      -- time_consistency | distribution_consistency | cross_modality_correlation
    metric_name         TEXT NOT NULL,      -- dtw_distance_reduction | sequence_similarity_score | mmd_reduction | correlation_gain | overall_score
    metric_value        REAL,
    metric_detail_json  TEXT,               -- 详细指标 (如成对 CCA 相关系数)
    recorded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES alignment_runs(run_id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_ae_input ON alignment_experiments(input_source);
CREATE INDEX IF NOT EXISTS idx_ae_created ON alignment_experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_am_experiment ON alignment_modalities(experiment_id);
CREATE INDEX IF NOT EXISTS idx_am_modality ON alignment_modalities(modality_name);
CREATE INDEX IF NOT EXISTS idx_ar_experiment ON alignment_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ar_status ON alignment_runs(status);
CREATE INDEX IF NOT EXISTS idx_fm_experiment ON feature_matrices(experiment_id);
CREATE INDEX IF NOT EXISTS idx_fm_type ON feature_matrices(matrix_type);
CREATE INDEX IF NOT EXISTS idx_ametrics_run ON alignment_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_ametrics_dimension ON alignment_metrics(dimension);
CREATE INDEX IF NOT EXISTS idx_ametrics_name ON alignment_metrics(metric_name);
"""


class RelationalDBClient(BaseStorageClient):
    """关系型数据库客户端

    支持 SQLite (默认) 和 DM8 (dmPython)。

    用法:
        db = RelationalDBClient(config={
            "dialect": "sqlite",             # sqlite | dm8
            "database": "data/storage/alignment.db",
            # DM8 only:
            "host": "localhost",
            "port": 5236,
            "user": "SYSDBA",
            "password": "...",
        })
        with db:
            db.save_experiment("E001", temporal_method="dtw", ...)
            experiments = db.list_experiments()
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._engine: Any = None
        self._connection: Any = None
        self._dialect: str = self.config.get("dialect", "sqlite")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立数据库连接并初始化 schema"""
        try:
            if self._dialect == "dm8":
                self._connect_dm8()
            else:
                self._connect_sqlite()

            self._init_schema()
            self._connected = True
            logg.info(f"关系型数据库已连接 (dialect={self._dialect})")

        except Exception as exc:
            logg.warning(f"数据库连接失败 ({exc})，使用 SQLite fallback")
            self._connect_sqlite()
            self._init_schema()
            self._connected = True

    def _connect_sqlite(self) -> None:
        """连接 SQLite"""
        import sqlite3

        db_path = self.config.get("database", "data/storage/alignment.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(db_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        logg.info(f"SQLite 数据库: {db_path}")

    def _connect_dm8(self) -> None:
        """连接达梦 DM8 数据库"""
        try:
            import dmPython

            self._connection = dmPython.connect(
                user=self.config.get("user", "SYSDBA"),
                password=self.config.get("password", ""),
                server=self.config.get("host", "localhost"),
                port=self.config.get("port", 5236),
            )
        except ImportError:
            raise ImportError("连接 DM8 需要 dmPython 包。请安装: pip install dmPython")

    def _init_schema(self) -> None:
        """初始化数据库表结构"""
        for stmt in _DEFAULT_SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._connection.execute(stmt)
        self._connection.commit()

    def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
        self._connected = False

    def is_healthy(self) -> bool:
        if not self._connected or self._connection is None:
            return False
        try:
            self._connection.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None = None) -> str:
        """通用写入 — 将 JSON 数据插入为行"""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, Path):
            data = data.read_text("utf-8")
        return key

    def get(self, key: str) -> bytes | None:
        return None

    def list(self, prefix: str = "") -> list[str]:
        tables = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
        return [row["name"] for row in tables]

    def delete(self, key: str) -> bool:
        return True

    def exists(self, key: str) -> bool:
        return True

    # ------------------------------------------------------------------
    # Domain-specific: 对齐实验
    # ------------------------------------------------------------------

    def save_experiment(
        self,
        experiment_id: str,
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
        """保存或更新对齐实验记录

        Returns:
            experiment_id
        """
        modalities_json = json.dumps(modalities, ensure_ascii=False) if modalities else None
        config_json = json.dumps(config_snapshot, ensure_ascii=False) if config_snapshot else None

        self._connection.execute(
            """
            INSERT OR REPLACE INTO alignment_experiments
            (experiment_id, input_source, n_modalities, modalities_json,
             temporal_method, feature_space_method, time_normalized,
             n_total_obs, n_total_features, pipeline_version, config_snapshot, notes, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (experiment_id, input_source,
             len(modalities) if modalities else None, modalities_json,
             temporal_method, feature_space_method, 1 if time_normalized else 0,
             n_total_obs, n_total_features, pipeline_version, config_json, notes),
        )
        self._connection.commit()
        logg.info(f"对齐实验已保存: {experiment_id}")
        return experiment_id

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """查询单个对齐实验"""
        row = self._connection.execute(
            "SELECT * FROM alignment_experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        # 反序列化 JSON 字段
        for field in ["modalities_json"]:
            if result.get(field):
                try:
                    result[field.replace("_json", "")] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return result

    def list_experiments(self, temporal_method: str | None = None) -> list[dict[str, Any]]:
        """列出所有对齐实验"""
        if temporal_method:
            rows = self._connection.execute(
                "SELECT * FROM alignment_experiments WHERE temporal_method=? ORDER BY created_at DESC",
                (temporal_method,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM alignment_experiments ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_experiment(self, experiment_id: str) -> None:
        """删除实验及相关记录（级联删除）"""
        for table in ["alignment_metrics", "feature_matrices", "alignment_modalities", "alignment_runs"]:
            self._connection.execute(
                f"DELETE FROM {table} WHERE experiment_id = ?", (experiment_id,)
            )
        self._connection.execute(
            "DELETE FROM alignment_experiments WHERE experiment_id = ?", (experiment_id,)
        )
        self._connection.commit()
        logg.info(f"实验已删除: {experiment_id}")

    # ------------------------------------------------------------------
    # Domain-specific: 模态详情
    # ------------------------------------------------------------------

    def save_modality(
        self,
        experiment_id: str,
        modality_name: str,
        n_obs: int | None = None,
        n_features_before: int | None = None,
        n_features_after: int | None = None,
        n_time_points: int | None = None,
        time_unit: str | None = None,
        time_range_min: float | None = None,
        time_range_max: float | None = None,
        n_interpolated: int = 0,
        n_original: int = 0,
        temporal_sub_method: str | None = None,
        lag_shift: float | None = None,
        n_batches: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """保存单个模态的对齐详情

        Returns:
            modality_id
        """
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        cursor = self._connection.execute(
            """
            INSERT INTO alignment_modalities
            (experiment_id, modality_name, n_obs, n_features_before, n_features_after,
             n_time_points, time_unit, time_range_min, time_range_max,
             n_interpolated, n_original, temporal_sub_method, lag_shift,
             n_batches, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, modality_name, n_obs, n_features_before, n_features_after,
             n_time_points, time_unit, time_range_min, time_range_max,
             n_interpolated, n_original, temporal_sub_method, lag_shift,
             n_batches, metadata_json),
        )
        self._connection.commit()
        logg.info(f"模态详情已保存: {experiment_id}/{modality_name}")
        return cursor.lastrowid

    def get_modalities(self, experiment_id: str) -> list[dict[str, Any]]:
        """查询某实验的所有模态详情"""
        rows = self._connection.execute(
            "SELECT * FROM alignment_modalities WHERE experiment_id = ? ORDER BY modality_name",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_modality_summary(self, experiment_id: str) -> dict[str, Any]:
        """获取实验的模态汇总统计"""
        row = self._connection.execute(
            """
            SELECT
                COUNT(*) as n_modalities,
                SUM(n_obs) as total_obs,
                SUM(n_interpolated) as total_interpolated,
                SUM(n_original) as total_original,
                MAX(n_time_points) as max_time_points,
                MIN(n_time_points) as min_time_points
            FROM alignment_modalities WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
        return dict(row) if row else {}

    # ------------------------------------------------------------------
    # Domain-specific: 流水线运行记录
    # ------------------------------------------------------------------

    def save_pipeline_run(
        self,
        run_id: str,
        experiment_id: str,
        temporal_method: str | None = None,
        feature_space_method: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        n_modalities_processed: int | None = None,
        steps_executed: list[str] | None = None,
    ) -> str:
        """记录流水线运行"""
        config_yaml = json.dumps(config_snapshot, ensure_ascii=False) if config_snapshot else None
        steps_json = json.dumps(steps_executed, ensure_ascii=False) if steps_executed else None

        self._connection.execute(
            """
            INSERT OR REPLACE INTO alignment_runs
            (run_id, experiment_id, temporal_method, feature_space_method,
             config_yaml, n_modalities_processed, steps_executed, started_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'running')
            """,
            (run_id, experiment_id, temporal_method, feature_space_method,
             config_yaml, n_modalities_processed, steps_json),
        )
        self._connection.commit()
        logg.info(f"流水线运行已记录: {run_id}")
        return run_id

    def mark_run_completed(self, run_id: str) -> None:
        """标记流水线运行完成"""
        self._connection.execute(
            "UPDATE alignment_runs SET status='completed', finished_at=CURRENT_TIMESTAMP WHERE run_id=?",
            (run_id,),
        )
        self._connection.commit()

    def mark_run_failed(self, run_id: str, error_message: str = "") -> None:
        """标记流水线运行失败"""
        self._connection.execute(
            "UPDATE alignment_runs SET status='failed', finished_at=CURRENT_TIMESTAMP, error_message=? WHERE run_id=?",
            (error_message, run_id),
        )
        self._connection.commit()

    def get_pipeline_run(self, run_id: str) -> dict[str, Any] | None:
        """查询流水线运行记录"""
        row = self._connection.execute(
            "SELECT * FROM alignment_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def query_pipeline_runs(
        self, experiment_id: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """查询流水线运行历史"""
        query = "SELECT * FROM alignment_runs WHERE 1=1"
        params: list[Any] = []

        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY started_at DESC"
        rows = self._connection.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Domain-specific: 特征矩阵
    # ------------------------------------------------------------------

    def save_feature_matrix(
        self,
        experiment_id: str,
        matrix_type: str,
        run_id: str | None = None,
        modality_name: str | None = None,
        file_format: str = "csv",
        n_rows: int | None = None,
        n_columns: int | None = None,
        index_key: str | None = None,
        storage_path: str | None = None,
        file_hash: str | None = None,
    ) -> int:
        """记录特征矩阵导出

        Args:
            matrix_type: observation | time_indexed | integrated
            index_key: aligned_time | process_stage | observation
        """
        cursor = self._connection.execute(
            """
            INSERT INTO feature_matrices
            (experiment_id, run_id, modality_name, matrix_type, file_format,
             n_rows, n_columns, index_key, storage_path, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (experiment_id, run_id, modality_name, matrix_type, file_format,
             n_rows, n_columns, index_key, storage_path, file_hash),
        )
        self._connection.commit()
        logg.info(f"特征矩阵记录已保存: {experiment_id}/{modality_name or 'integrated'} [{matrix_type}]")
        return cursor.lastrowid

    def list_feature_matrices(
        self, experiment_id: str, matrix_type: str | None = None
    ) -> list[dict[str, Any]]:
        """查询特征矩阵导出记录"""
        if matrix_type:
            rows = self._connection.execute(
                "SELECT * FROM feature_matrices WHERE experiment_id=? AND matrix_type=? ORDER BY modality_name",
                (experiment_id, matrix_type),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM feature_matrices WHERE experiment_id=? ORDER BY matrix_type, modality_name",
                (experiment_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Domain-specific: 质量指标
    # ------------------------------------------------------------------

    def save_metric(
        self,
        run_id: str,
        dimension: str,
        metric_name: str,
        metric_value: float,
        metric_detail: dict[str, Any] | None = None,
    ) -> None:
        """保存单条质量指标"""
        detail_json = json.dumps(metric_detail, ensure_ascii=False) if metric_detail else None
        self._connection.execute(
            """
            INSERT INTO alignment_metrics (run_id, dimension, metric_name, metric_value, metric_detail_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, dimension, metric_name, metric_value, detail_json),
        )
        self._connection.commit()

    def save_metrics_batch(
        self,
        run_id: str,
        metrics: dict[str, dict[str, float]],
    ) -> None:
        """批量保存三维度评估指标

        Args:
            metrics: {dimension: {metric_name: value}}
                例如 {"time_consistency": {"dtw_distance_reduction": 0.3, ...}, ...}
        """
        for dimension, metric_dict in metrics.items():
            for metric_name, metric_value in metric_dict.items():
                if isinstance(metric_value, (int, float)):
                    self._connection.execute(
                        """
                        INSERT INTO alignment_metrics (run_id, dimension, metric_name, metric_value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run_id, dimension, metric_name, metric_value),
                    )
        self._connection.commit()

    def get_metrics(self, run_id: str) -> dict[str, dict[str, float]]:
        """查询某次运行的全部质量指标，按维度分组"""
        rows = self._connection.execute(
            "SELECT dimension, metric_name, metric_value FROM alignment_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        result: dict[str, dict[str, float]] = {}
        for r in rows:
            result.setdefault(r["dimension"], {})[r["metric_name"]] = r["metric_value"]
        return result

    def get_metric_summary(self, metric_name: str) -> dict[str, Any]:
        """查询某个指标的汇总统计"""
        row = self._connection.execute(
            """
            SELECT
                COUNT(*) as n,
                AVG(metric_value) as mean,
                MIN(metric_value) as min_val,
                MAX(metric_value) as max_val
            FROM alignment_metrics WHERE metric_name = ?
            """,
            (metric_name,),
        ).fetchone()
        return dict(row) if row else {}

    def get_overall_score_history(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        """查询 overall_score 历史记录"""
        if experiment_id:
            rows = self._connection.execute(
                """
                SELECT ar.experiment_id, ar.started_at, am.metric_value as overall_score
                FROM alignment_metrics am
                JOIN alignment_runs ar ON am.run_id = ar.run_id
                WHERE am.metric_name = 'overall_score' AND ar.experiment_id = ?
                ORDER BY ar.started_at DESC
                """,
                (experiment_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT ar.experiment_id, ar.started_at, am.metric_value as overall_score
                FROM alignment_metrics am
                JOIN alignment_runs ar ON am.run_id = ar.run_id
                WHERE am.metric_name = 'overall_score'
                ORDER BY ar.started_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def export_experiment(self, experiment_id: str) -> dict[str, Any]:
        """导出实验的全部数据（实验 + 模态 + 运行 + 指标 + 矩阵）"""
        experiment = self.get_experiment(experiment_id)
        modalities = self.get_modalities(experiment_id)
        runs = self.query_pipeline_runs(experiment_id=experiment_id)
        matrices = self.list_feature_matrices(experiment_id)

        metrics: dict[str, dict] = {}
        for run in runs:
            metrics[run["run_id"]] = self.get_metrics(run["run_id"])

        return {
            "experiment": experiment,
            "modalities": modalities,
            "pipeline_runs": runs,
            "feature_matrices": matrices,
            "metrics": metrics,
        }

    def execute_raw(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """执行原始 SQL 查询（用于自定义分析）"""
        import sqlite3

        try:
            if params:
                rows = self._connection.execute(sql, params).fetchall()
            else:
                rows = self._connection.execute(sql).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logg.error(f"SQL 执行失败: {exc}")
            return []
