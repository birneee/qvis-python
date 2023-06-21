from __future__ import annotations

from datetime import timedelta
from typing import Iterator

from .event_iterator import EventIterator
from .frame_iterator import FrameIterator


class PacketIterator:
    event: EventIterator

    def __init__(self, event: EventIterator):
        self.event = event

    @property
    def time(self) -> float:
        """in ms"""
        return self.event.time

    @property
    def time_as_timedelta(self) -> timedelta:
        return self.event.time_as_timedelta

    @property
    def header(self) -> dict:
        return self.event.data.get('header')

    @property
    def packet_number(self) -> int:
        return self.header.get('packet_number')

    @property
    def raw_length(self) -> int:
        return self.event.data['raw']['length']

    @property
    def frames(self) -> Iterator['FrameIterator']:
        data = self.event.data
        if data is None:
            return
        frames = data.get('frames')
        if frames is None:
            return
        for frame in frames:
            yield FrameIterator(frame, self)

    def frames_of_type(self, frame_type: str) -> Iterator['FrameIterator']:
        return filter(lambda f: f.type == frame_type, self.frames)
