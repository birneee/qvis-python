from qvis import frame_types
from qvis.event import T_DICT
from qvis.frame import Frame


class DatagramFrame(Frame[T_DICT]):
    def __init__(self, base: Frame[T_DICT]):
        super().__init__(base.inner, base.packet)
        if self.type != frame_types.DATAGRAM:
            raise Exception("invalid frame_type")

    @property
    def length(self) -> int:
        return int(self.inner['length'])
