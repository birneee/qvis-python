from qvis import frame_types
from qvis.frame import Frame


class DatagramFrame(Frame):
    def __init__(self, base: Frame):
        super().__init__(base.inner, base.packet)
        if self.type != frame_types.DATAGRAM:
            raise Exception("invalid frame_type")

    @property
    def length(self) -> int:
        return int(self.inner['length'])
