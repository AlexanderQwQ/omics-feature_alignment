"""配置管理系统"""

from ._settings import Settings
from ._verbosity import Verbosity

# 全局单例
settings = Settings()

__all__ = ["Settings", "Verbosity", "settings"]
