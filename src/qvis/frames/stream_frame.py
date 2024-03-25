from qvis.frame import Frame


class StreamFrame(Frame):
    base: Frame

    def __init__(self, base: Frame):
        super().__init__(base.inner, base.packet)
        self.base = base

    @property
    def stream_id(self) -> int:
        return self.base.inner.get('stream_id')

    @property
    def offset(self) -> int:
        return self.base.inner.get('offset')

    @property
    def length(self) -> int:
        return self.base.inner.get('length')

    @property
    def fin(self) -> bool:
        return self.base.inner.get('fin')
