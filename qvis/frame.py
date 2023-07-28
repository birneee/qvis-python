from __future__ import annotations
from datetime import timedelta
from typing import Generic, TYPE_CHECKING

from qvis.event import T_DICT

if TYPE_CHECKING:
    from qvis.packet import Packet


class Frame(Generic[T_DICT]):
    inner: T_DICT
    packet: Packet[T_DICT]

    def __init__(self, inner: T_DICT, packet: Packet[T_DICT]):
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

    def __getitem__(self, item):
        return self.inner[item]
