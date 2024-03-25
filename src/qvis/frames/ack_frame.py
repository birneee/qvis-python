from qvis.frame import Frame
from qvis.ranges import Ranges


class AckFrame(Frame):
    base: Frame

    def __init__(self, base: Frame):
        super().__init__(base.inner, base.packet)
        self.base = base

    @property
    def acked_packet_numbers(self) -> Ranges:
        return Ranges(self.base.inner.get("acked_ranges"))
