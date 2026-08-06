"""兼容性模块，处理不同依赖版本之间的 API 差异"""

from __future__ import annotations

from functools import partial, update_wrapper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def _copy_docs_and_signature(fn: Callable):
    """将源函数的 docstring 和注解复制到目标函数"""
    return partial(update_wrapper, wrapped=fn, assigned=["__doc__", "__annotations__"])
