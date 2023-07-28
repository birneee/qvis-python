from qvis.frame import Frame


class MaxStreamDataFrame:
    base: Frame

    def __init__(self, base: Frame):
        self.base = base

    @property
    def stream_id(self) -> int:
        return self.base.inner.get('stream_id')

    @property
    def maximum(self) -> int:
        return self.base.inner.get('maximum')
