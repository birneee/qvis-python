import time
from typing import Callable


def print_func_time(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f'ran {func.__name__} in {end - start} s')
        return res
    if hasattr(func, '__name__'):
        wrapper.__name__ = func.__name__
    else:
        wrapper.__name__ = func.fget.__name__
    return wrapper
