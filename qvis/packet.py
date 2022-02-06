from __future__ import annotations
from typing import Iterator, Optional

from . import frame_types
from .event import Event
from .frame import Frame, StreamFrame


class Packet:
    event: Event

    def __init__(self, event: Event):
        self.event = event

    @property
    def time(self) -> float:
        """in seconds"""
        return self.event.time

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
    def frames(self) -> Iterator['Frame']:
        data = self.event.data
        if data is None:
            return
        frames = data.get('frames')
        if frames is None:
            return
        for frame in frames:
            yield Frame(frame, self)

    def frames_of_type(self, frame_type: str) -> Iterator['Frame']:
        return filter(lambda f: f.type == frame_type, self.frames)

    @property
    def stream_frames(self) -> Iterator['StreamFrame']:
        return map(lambda f: StreamFrame(f), self.frames_of_type(frame_types.STREAM))

    def stream_frames_of_stream(self, stream_id: int) -> Iterator['StreamFrame']:
        return filter(lambda f: f.stream_id == stream_id, self.stream_frames)
