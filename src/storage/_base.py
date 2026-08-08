"""
混合存储客户端抽象基类

定义统一的 put/get/list/delete/query 接口，
确保三种存储后端（对象存储、关系型、图数据库）行为一致。

与 Module 1 (omics_standardization) 的 storage._base 保持接口兼容。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class BaseStorageClient(ABC):
    """存储客户端抽象基类

    所有存储后端（MinIO, SQLite/DM8, Neo4j）统一实现此接口。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._connected = False

    @abstractmethod
    def connect(self) -> None:
        """建立连接"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        ...

    @abstractmethod
    def is_healthy(self) -> bool:
        """健康检查"""
        ...

    # ---------------------------------------------------------------
    # CRUD 接口
    # ---------------------------------------------------------------

    @abstractmethod
    def put(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None = None) -> str:
        """写入对象/记录

        Args:
            key: 唯一标识符 (MinIO: object key, SQL: primary key)
            data: 数据内容 (bytes, str, 或本地文件路径)
            metadata: 附加元数据 (MinIO: object tags, SQL: extra columns)

        Returns:
            存储后的标识符 (MinIO: etag, SQL: row id)
        """
        ...

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """读取对象/记录"""
        ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """列出所有匹配前缀的键"""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除对象/记录"""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """检查是否存在"""
        ...

    def __enter__(self) -> BaseStorageClient:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.disconnect()
