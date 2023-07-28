from enum import Enum

from qvis.perspective import Perspective


class StreamType(Enum):
    UNI = 1
    BIDI = 2


class StreamID:
    inner: int

    def __init__(self, stream_id: int):
        self.inner = stream_id

    def initiated_by(self) -> Perspective:
        if self.inner % 2 == 0:
            return Perspective.CLIENT
        else:
            return Perspective.SERVER

    def type(self) -> StreamType:
        if self.inner % 4 >= 2:
            return StreamType.UNI
        else:
            return StreamType.BIDI
