import os
from typing import Callable, Optional, Any


class PropertyMemoryCache:
    dict: dict[str, any]

    def __init__(self):
        self.dict = {}
        pass

    def load(self, name: str) -> Optional[Any]:
        return self.dict.get(name)

    def store(self, name: str, value: Any):
        self.dict[name] = value


def property_memory_cache(func: Callable) -> Callable:
    if os.environ.get('NOCACHE') == '1':
        return func

    def wrapper(*args, **kwargs):
        if len(args) != 1:
            raise Exception(f'parameters are not allowed')
        cache: PropertyMemoryCache = args[0].__property_memory_cache__
        value = cache.load(func.__name__)
        if value is not None:
            print(f'load {func.__name__} from memory cache')
            return value
        value = func(*args, **kwargs)
        cache.store(func.__name__, value)
        return value

    wrapper.__name__ = func.__name__
    return wrapper
