from qvis import frame_types
from qvis.frame import Frame
from qvis.frame_iterator import FrameIterator


class DatagramFrameIterator(FrameIterator):
    def __init__(self, base: FrameIterator):
        super().__init__(base.inner, base.packet)
        if self.type != frame_types.DATAGRAM:
            raise Exception("invalid frame_type")

    @property
    def length(self) -> int:
        return int(self.inner['length'])
