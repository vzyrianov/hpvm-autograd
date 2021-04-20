import logging
import os
from logging import config
from pathlib import Path
from typing import Union

import tqdm

PathLike = Union[Path, str]


class TqdmStreamHandler(logging.Handler):
    """tqdm-friendly logging handler. Uses tqdm.write instead of print for logging."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except:
            self.handleError(record)


# noinspection PyTypeChecker
_last_applied_config: dict = None


def config_pylogger(
    filename: str = None, output_dir: PathLike = None, verbose: bool = False
) -> logging.Logger:
    """Configure the Python logger.

    For each execution of the application, we'd like to create a unique log file.
    By default this file is named using the date and time of day, so that it can be sorted by recency.
    You can also name your filename or choose the log directory.
    """
    import time

    timestr = time.strftime("%Y.%m.%d-%H%M%S")
    filename = filename or timestr
    output_dir = Path(output_dir or ".")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = output_dir / filename

    global _last_applied_config
    _last_applied_config = d = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "%(levelname)s %(name)s: " "%(message)s"},
            "detailed": {
                "format": "[%(asctime)-15s] "
                "%(levelname)7s %(name)s: "
                "%(message)s "
                "@%(filename)s:%(lineno)d"
            },
        },
        "handlers": {
            "console": {
                "()": TqdmStreamHandler,
                "level": "INFO",
                "formatter": "simple",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": file_path.as_posix(),
                "mode": "a",  # Because we may apply this config again, want to keep existing content
                "formatter": "detailed",
            },
        },
        "root": {
            "level": "DEBUG" if verbose else "INFO",
            "handlers": ["console", "file"],
        },
    }
    config.dictConfig(d)

    msglogger = logging.getLogger()
    msglogger.info(f"Log file for this run: {file_path}")
    return msglogger


def override_opentuner_config():
    if _last_applied_config is not None:
        config.dictConfig(_last_applied_config)
    if Path("opentuner.log").is_file():
        Path("opentuner.log").unlink()
