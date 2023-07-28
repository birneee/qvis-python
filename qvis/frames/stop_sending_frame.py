from qvis.frame import Frame


class StopSendingFrame(Frame):
    base: Frame

    def __init__(self, base: Frame):
        super().__init__(base.inner, base.packet)
        self.base = base

    @property
    def stream_id(self) -> int:
        return self.base.inner.get('stream_id')
