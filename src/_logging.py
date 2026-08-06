"""
日志系统（scanpy 风格）

提供与 settings.verbosity 集成的自定义 RootLogger，
以及 info / warning / error / hint / debug 便捷函数。
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING

# 在 INFO 和 DEBUG 之间插入 HINT 级别
HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, "HINT")


class _RootLogger(logging.RootLogger):
    """自定义根日志记录器，支持 times 和 deep 参数"""

    def __init__(self, level: int) -> None:
        super().__init__(level)
        self.propagate = False
        _RootLogger.manager = logging.Manager(self)

    def log(
        self,
        level: int,
        msg: str,
        *,
        extra: dict | None = None,
        time: datetime | None = None,
        deep: str | None = None,
    ) -> datetime:
        from _settings import settings

        now = datetime.now(timezone.utc)
        time_passed: timedelta | None = None if time is None else now - time
        extra = {
            **(extra or {}),
            "deep": deep if settings.verbosity.level < level else None,
            "time_passed": time_passed,
        }
        super().log(level, msg, extra=extra)
        return now

    def critical(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(CRITICAL, msg, time=time, deep=deep, extra=extra)

    def error(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(ERROR, msg, time=time, deep=deep, extra=extra)

    def warning(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(WARNING, msg, time=time, deep=deep, extra=extra)

    def info(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(INFO, msg, time=time, deep=deep, extra=extra)

    def hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(HINT, msg, time=time, deep=deep, extra=extra)

    def debug(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(DEBUG, msg, time=time, deep=deep, extra=extra)


class _LogFormatter(logging.Formatter):
    """自定义日志格式器"""

    def __init__(
        self, fmt: str = "{levelname}: {message}", datefmt: str = "%Y-%m-%d %H:%M", style: str = "{"
    ) -> None:
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        format_orig = self._style._fmt
        if record.levelno == INFO:
            self._style._fmt = "{message}"
        elif record.levelno == HINT:
            self._style._fmt = "--> {message}"
        elif record.levelno == DEBUG:
            self._style._fmt = "    {message}"
        if record.time_passed:
            if record.time_passed.microseconds:
                record.time_passed = timedelta(seconds=int(record.time_passed.total_seconds()))
            if "{time_passed}" not in record.msg:
                self._style._fmt += " ({time_passed})"
        if record.deep:
            record.msg = f"{record.msg}: {record.deep}"
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


# 便捷函数（供内部模块使用）


def _emit(level: int, msg: str, *, time=None, deep=None, extra=None) -> datetime:
    """Core logging function — writes directly to the omics_align logger."""
    from _settings import settings

    logger = logging.getLogger("omics_align")
    now = datetime.now(timezone.utc)
    time_passed = None if time is None else now - time
    show_deep = int(settings.verbosity) >= 3
    extra_dict = {
        **(extra or {}),
        "deep": deep if show_deep else None,
        "time_passed": time_passed,
    }
    logger._log(level, msg, (), extra=extra_dict)
    return now


def error(msg: str, *, time=None, deep=None, extra=None) -> datetime:
    return _emit(ERROR, msg, time=time, deep=deep, extra=extra)


def warning(msg: str, *, time=None, deep=None, extra=None) -> datetime:
    return _emit(WARNING, msg, time=time, deep=deep, extra=extra)


def info(msg: str, *, time=None, deep=None, extra=None) -> datetime:
    return _emit(INFO, msg, time=time, deep=deep, extra=extra)


def hint(msg: str, *, time=None, deep=None, extra=None) -> datetime:
    return _emit(HINT, msg, time=time, deep=deep, extra=extra)


def debug(msg: str, *, time=None, deep=None, extra=None) -> datetime:
    return _emit(DEBUG, msg, time=time, deep=deep, extra=extra)
