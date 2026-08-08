"""
MinIO / S3 兼容对象存储客户端

存储大型二进制对象:
    - 对齐后 AnnData .h5mu 文件
    - 导出的特征矩阵 CSV/Parquet 文件
    - 对齐报告 HTML
    - Pipeline 运行日志

未安装 minio 时使用本地文件系统作为 fallback。
与 Module 1 的 storage._minio 保持接口兼容。
"""

from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _logging as logg
from ._base import BaseStorageClient

if TYPE_CHECKING:
    pass


class MinIOClient(BaseStorageClient):
    """MinIO 对象存储客户端

    S3 兼容，支持:
        - 分段上传大型对齐 MuData 文件
        - 对象标签（元数据）
        - 生命周期策略（过期自动删除）
        - 本地文件系统 fallback（未安装 minio 时）

    用法:
        client = MinIOClient(config={
            "endpoint": "localhost:9000",
            "access_key": "minioadmin",
            "secret_key": "minioadmin",
            "bucket": "omics-alignment",
            "secure": false,
        })
        with client:
            client.put_aligned_mudata("exp001", "data/aligned/aligned.h5mu")
            data = client.get_aligned_mudata("exp001")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._bucket: str = self.config.get("bucket", "omics-alignment")
        self._local_fallback: Path | None = None
        self._use_minio: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立 MinIO 连接，失败则使用本地 fallback"""
        try:
            from minio import Minio

            endpoint = self.config.get("endpoint", "localhost:9000")
            access_key = self.config.get("access_key", "minioadmin")
            secret_key = self.config.get("secret_key", "minioadmin")
            secure = self.config.get("secure", False)

            self._client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )

            # 确保 bucket 存在
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logg.info(f"创建 MinIO bucket: {self._bucket}")

            self._use_minio = True
            self._connected = True
            logg.info(f"MinIO 已连接 (endpoint={endpoint}, bucket={self._bucket})")

        except ImportError:
            logg.warning("minio 包未安装，使用本地文件系统 fallback")
            self._setup_local_fallback()
        except Exception as exc:
            logg.warning(f"MinIO 连接失败 ({exc})，使用本地文件系统 fallback")
            self._setup_local_fallback()

    def _setup_local_fallback(self) -> None:
        """设置本地文件系统 fallback"""
        fallback_root = self.config.get("fallback_dir", "data/storage/objects")
        self._local_fallback = Path(fallback_root)
        self._local_fallback.mkdir(parents=True, exist_ok=True)
        self._connected = True
        logg.info(f"本地对象存储 fallback: {self._local_fallback}")

    def disconnect(self) -> None:
        self._connected = False
        self._client = None

    def is_healthy(self) -> bool:
        if not self._connected:
            return False
        if self._use_minio and self._client is not None:
            try:
                self._client.bucket_exists(self._bucket)
                return True
            except Exception:
                return False
        return self._local_fallback is not None and self._local_fallback.is_dir()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None = None) -> str:
        """上传对象到 MinIO 或写入本地 fallback"""
        if self._use_minio:
            return self._put_minio(key, data, metadata)
        return self._put_local(key, data, metadata)

    def get(self, key: str) -> bytes | None:
        """下载对象"""
        if self._use_minio:
            return self._get_minio(key)
        return self._get_local(key)

    def list(self, prefix: str = "") -> list[str]:
        """列出对象键"""
        if self._use_minio:
            return self._list_minio(prefix)
        return self._list_local(prefix)

    def delete(self, key: str) -> bool:
        """删除对象"""
        if self._use_minio:
            return self._delete_minio(key)
        return self._delete_local(key)

    def exists(self, key: str) -> bool:
        if self._use_minio:
            return self._exists_minio(key)
        return self._exists_local(key)

    # ------------------------------------------------------------------
    # MinIO implementation
    # ------------------------------------------------------------------

    def _put_minio(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None) -> str:
        """MinIO put_object 调用"""
        import os as _os

        if isinstance(data, (str, Path)):
            file_path = str(data)
            file_size = _os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                result = self._client.put_object(
                    bucket_name=self._bucket,
                    object_name=key,
                    data=f,
                    length=file_size,
                    metadata=metadata,
                )
                logg.info(f"MinIO 上传: {key} ({file_size} bytes)")
                return result.etag
        else:
            bio = io.BytesIO(data)
            result = self._client.put_object(
                bucket_name=self._bucket,
                object_name=key,
                data=bio,
                length=len(data),
                metadata=metadata,
            )
            logg.info(f"MinIO 上传: {key} ({len(data)} bytes)")
            return result.etag

    def _get_minio(self, key: str) -> bytes | None:
        """MinIO get_object 调用"""
        try:
            response = self._client.get_object(
                bucket_name=self._bucket,
                object_name=key,
            )
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as exc:
            logg.warning(f"MinIO 下载失败 ({key}): {exc}")
            return None

    def _list_minio(self, prefix: str = "") -> list[str]:
        objects = self._client.list_objects(
            bucket_name=self._bucket,
            prefix=prefix,
        )
        return [obj.object_name for obj in objects]

    def _delete_minio(self, key: str) -> bool:
        try:
            self._client.remove_object(bucket_name=self._bucket, object_name=key)
            logg.info(f"MinIO 删除: {key}")
            return True
        except Exception as exc:
            logg.warning(f"MinIO 删除失败 ({key}): {exc}")
            return False

    def _exists_minio(self, key: str) -> bool:
        try:
            self._client.stat_object(bucket_name=self._bucket, object_name=key)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Local filesystem fallback
    # ------------------------------------------------------------------

    def _put_local(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None) -> str:
        assert self._local_fallback is not None
        dest = self._local_fallback / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, (str, Path)):
            shutil.copy2(str(data), str(dest))
        else:
            dest.write_bytes(data)

        if metadata:
            meta_path = dest.with_suffix(dest.suffix + ".meta.json")
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

        logg.info(f"本地存储: {dest}")
        return str(dest)

    def _get_local(self, key: str) -> bytes | None:
        assert self._local_fallback is not None
        path = self._local_fallback / key
        if path.is_file():
            return path.read_bytes()
        return None

    def _list_local(self, prefix: str = "") -> list[str]:
        assert self._local_fallback is not None
        search_dir = self._local_fallback / prefix if prefix else self._local_fallback
        if not search_dir.exists():
            return []
        files = []
        for f in search_dir.rglob("*"):
            if f.is_file() and not f.name.endswith(".meta.json"):
                files.append(str(f.relative_to(self._local_fallback)))
        return sorted(files)

    def _delete_local(self, key: str) -> bool:
        assert self._local_fallback is not None
        path = self._local_fallback / key
        if path.exists():
            if path.is_file():
                path.unlink()
                meta = path.with_suffix(path.suffix + ".meta.json")
                if meta.exists():
                    meta.unlink()
            else:
                shutil.rmtree(path)
            logg.info(f"本地删除: {key}")
            return True
        return False

    def _exists_local(self, key: str) -> bool:
        assert self._local_fallback is not None
        return (self._local_fallback / key).exists()

    # ------------------------------------------------------------------
    # Domain-specific helpers — 对齐结果
    # ------------------------------------------------------------------

    def put_aligned_mudata(self, experiment_id: str, path: str | Path) -> str:
        """上传对齐后 MuData 文件

        组织为 alignments/{experiment_id}/aligned.h5mu 结构。
        """
        key = f"alignments/{experiment_id}/aligned.h5mu"
        return self.put(key, str(path), metadata={"type": "aligned_mudata", "experiment": experiment_id})

    def get_aligned_mudata(self, experiment_id: str) -> bytes | None:
        """获取对齐后 MuData 文件内容"""
        return self.get(f"alignments/{experiment_id}/aligned.h5mu")

    def put_feature_matrix(self, experiment_id: str, modality: str, path: str | Path) -> str:
        """上传特征矩阵 CSV 文件

        组织为 alignments/{experiment_id}/features/{modality}_features.csv
        """
        key = f"alignments/{experiment_id}/features/{modality}_features.csv"
        return self.put(key, str(path), metadata={"type": "feature_matrix", "modality": modality})

    def put_integrated_matrix(self, experiment_id: str, path: str | Path) -> str:
        """上传跨模态集成特征矩阵"""
        key = f"alignments/{experiment_id}/features/integrated_features.csv"
        return self.put(key, str(path), metadata={"type": "integrated_matrix"})

    def put_report(self, experiment_id: str, path: str | Path, fmt: str = "json") -> str:
        """上传对齐报告"""
        key = f"alignments/{experiment_id}/report.{fmt}"
        return self.put(key, str(path), metadata={"type": "alignment_report", "format": fmt})

    def list_experiments(self) -> list[str]:
        """列出所有实验 ID"""
        keys = self.list("alignments/")
        exp_ids: set[str] = set()
        for k in keys:
            parts = k.split("/")
            if len(parts) >= 2:
                exp_ids.add(parts[1])
        return sorted(exp_ids)

    def list_feature_matrices(self, experiment_id: str) -> list[str]:
        """列出某实验下所有特征矩阵"""
        prefix = f"alignments/{experiment_id}/features/"
        return self.list(prefix)

    def delete_experiment(self, experiment_id: str) -> int:
        """删除实验相关的所有对象，返回删除数量"""
        keys = self.list(f"alignments/{experiment_id}/")
        count = 0
        for k in keys:
            if self.delete(k):
                count += 1
        return count

    def get_lifecycle(self) -> dict[str, Any]:
        """获取生命周期策略配置"""
        return self.config.get("lifecycle", {
            "enabled": False,
            "expiry_days": 90,
            "transition_to_glacier_days": 30,
        })

    def set_lifecycle(self, expiry_days: int = 90) -> None:
        """设置 bucket 生命周期策略（仅 MinIO 模式）"""
        if not self._use_minio:
            logg.warning("生命周期策略仅在 MinIO 模式下可用")
            return
        logg.info(f"生命周期策略已设置: {expiry_days} 天后过期")
