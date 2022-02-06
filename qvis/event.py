from typing import Optional


class Event:
    inner: dict

    def __init__(self, inner: dict):
        self.inner = inner

    @property
    def index(self) -> int:
        """unique index of the event"""
        return getattr(self.inner, 'Index')

    @property
    def time(self) -> float:
        """in ms"""
        return self.inner['time']

    @property
    def name(self) -> str:
        """name of the event"""
        return self.inner['name']

    @property
    def data(self) -> Optional[dict]:
        """event specific data"""
        return self.inner.get('data')


class XseRecord:
    inner: Event

    def __init__(self, event: Event):
        self.inner = event

    @property
    def time(self) -> float:
        """in ms"""
        return self.inner.time

    @property
    def stream_id(self) -> int:
        return self.inner.data.get('stream_id')

    @property
    def raw_length(self) -> int:
        return self.inner.data.get('raw_length')

    @property
    def data_length(self) -> int:
        return self.inner.data.get('data_length')
