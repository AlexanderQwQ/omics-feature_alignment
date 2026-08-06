"""日志详细级别枚举"""

from __future__ import annotations

from enum import IntEnum


class Verbosity(IntEnum):
    """日志详细级别

    值从低到高：error < warning < info < hint < debug
    """

    error = 0
    warning = 1
    info = 2
    hint = 3
    debug = 4

    def __str__(self) -> str:
        return self.name
