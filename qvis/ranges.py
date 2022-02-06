from __future__ import annotations
import bisect
from typing import Iterator


def ranges_to_set(ranges: list[list[int]]) -> set[int]:
    _set = set()
    for _range in ranges:
        if len(_range) == 1:
            _set.add(_range[0])
        else:
            for elem in range(_range[0], _range[1] + 1):
                _set.add(elem)
    return _set


class Ranges:
    inner: list[range]

    @property
    def lowest(self) -> int | None:
        if len(self.inner) == 0:
            return None
        return self.inner[0].start

    @property
    def largest(self) -> int | None:
        if len(self.inner) == 0:
            return None
        return self.inner[-1].stop - 1

    @property
    def has_missing_ranges(self) -> bool:
        return len(self.inner) >= 2

    def __getitem__(self, item) -> range:
        return self.inner[item]

    def __len__(self) -> int:
        return len(self.inner)

    def __init__(self, ranges: list[list[int]] | list[range]):
        if len(ranges) == 0:
            self.inner = []
            return
        if isinstance(ranges[0], range):
            self.inner = ranges
            self.inner.sort(key=lambda r: r.start)
            return
        if isinstance(ranges[0], list):
            self.inner = []
            for _range in ranges:
                if len(_range) == 1:
                    bisect.insort(self.inner, range(_range[0], _range[0] + 1), key=lambda r: r.start)
                else:
                    bisect.insort(self.inner, range(_range[0], _range[1] + 1), key=lambda r: r.start)
            return
        raise "invalid type"

    def containing_range(self, elem: int) -> tuple[int, range] | tuple[None, None]:
        for index, _range in enumerate(self.inner):
            if elem < _range.start:
                continue
            if elem >= _range.stop:
                return None, None
            return index, _range
        return None, None

    def __contains__(self, elem: int) -> bool:
        for _range in self.inner:
            if elem < _range.start:
                continue
            if elem >= _range.stop:
                return False
            return True
        return False

    def iterate_elements(self) -> Iterator[int]:
        for _range in self.inner:
            yield from _range

    def iterate_elements_reversed(self):
        for _range in reversed(self.inner):
            yield from reversed(_range)
