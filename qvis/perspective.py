from enum import Enum
from typing_extensions import Self


class Perspective(Enum):
    CLIENT = 1
    SERVER = 2

    @classmethod
    def from_str(cls, str) -> Self:
        if str == 'client':
            return cls.CLIENT
        elif str == 'server':
            return cls.SERVER
        raise RuntimeError('unexpected value')
