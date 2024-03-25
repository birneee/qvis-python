from itertools import islice
from typing import Iterable, TypeVar

K = TypeVar("K")


def window(seq: Iterable[K], n=2) -> Iterable[tuple[K]]:
    "Source: https://stackoverflow.com/a/6822773/9401943"
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result