import io
import os
from pathlib import Path
from typing import Callable, Optional, Any

import msgpack


class Cache:
    folder_path: Path
    qlog_file: Path

    def __init__(self, qlog_file: Path):
        self.qlog_file = qlog_file
        self.folder_path = qlog_file.parent.joinpath(f'{qlog_file.name}.index')

    def load(self, name: str) -> Optional[Any]:
        path = self.folder_path.joinpath(name)
        if not path.exists():
            return None
        if path.stat().st_mtime < self.qlog_file.stat().st_mtime:
            return None
        with io.open(path, "rb") as f:
            return msgpack.unpackb(f.read())

    def store(self, name: str, value: Any):
        self.folder_path.mkdir(parents=True, exist_ok=True)
        path = self.folder_path.joinpath(name)
        with io.open(path, "wb") as f:
            f.write(msgpack.packb(value))


def autocache(func: Callable) -> Callable:
    if os.environ.get('NOCACHE') == '1':
        return func

    def wrapper(*args, **kwargs):
        if len(args) != 1:
            raise Exception(f'parameters are not allowed')
        cache: Cache = args[0].__cache__
        value = cache.load(func.__name__)
        if value != None:
            print(f'load {func.__name__} from disk cache')
            return value
        value = func(*args, **kwargs)
        cache.store(func.__name__, value)
        print(f'stored {func.__name__} in disk cache')
        return value
    if hasattr(func, '__name__'):
        wrapper.__name__ = func.__name__
    else:
        wrapper.__name__ = func.fget.__name__
    return wrapper
