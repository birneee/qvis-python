from datetime import timedelta

from cysimdjson import JSONObject


class FrameIterator(object):
    inner: JSONObject
    packet: 'packet.Packet'

    def __init__(self, inner: JSONObject, packet: 'packet.Packet'):
        self.inner = inner
        self.packet = packet

    @property
    def type(self) -> str:
        return self.inner['frame_type']

    @property
    def time(self) -> float:
        """in ms"""
        return self.packet.time

    @property
    def time_as_timedelta(self) -> timedelta:
        return self.packet.time_as_timedelta
