from typing import Optional, Iterator

from qvis.ranges import Ranges


class Frame(object):
    inner: dict
    packet: 'packet.Packet'

    def __init__(self, inner: dict, packet: 'packet.Packet'):
        self.inner = inner
        self.packet = packet

    @property
    def type(self) -> str:
        return self.inner['frame_type']

    @property
    def time(self) -> float:
        """in seconds"""
        return self.packet.time


class AckFrame(Frame):
    base: Frame

    def __init__(self, base: Frame):
        super().__init__(base.inner, base.packet)
        self.base = base

    @property
    def acked_packet_numbers(self) -> Ranges:
        return Ranges(self.base.inner.get("acked_ranges"))


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
