from __future__ import annotations

from datetime import timedelta

from typing import Iterator, Generic, TYPE_CHECKING

from . import frame_types
from .event import Event, T_DICT

if TYPE_CHECKING:
    from .frame import Frame
    from .frames.stream_frame import StreamFrame


class PacketHeader(Generic[T_DICT]):
    inner: T_DICT

    def __init__(self, inner: T_DICT):
        self.inner = inner

    @property
    def packet_type(self) -> str:
        return self['packet_type']

    @property
    def packet_number(self) -> int:
        return self['packet_number']

    def __getitem__(self, item):
        return self.inner[item]


class Packet(Generic[T_DICT]):
    event: Event[T_DICT]

    def __init__(self, event: Event[T_DICT]):
        self.event = event

    @property
    def time(self) -> float:
        """in ms"""
        return self.event.time

    @property
    def time_as_timedelta(self) -> timedelta:
        return self.event.time_as_timedelta

    @property
    def header(self) -> PacketHeader[T_DICT]:
        return PacketHeader(self.event.data.get('header'))

    @property
    def packet_number(self) -> int:
        return self.header.packet_number

    @property
    def raw_length(self) -> int:
        return self.event.data['raw']['length']

    @property
    def frames(self) -> Iterator[Frame[T_DICT]]:
        from .frame import Frame
        data = self.event.data
        if data is None:
            return
        frames = data.get('frames')
        if frames is None:
            return
        for frame in frames:
            yield Frame(frame, self)

    def frames_of_type(self, frame_type: str) -> Iterator[Frame[T_DICT]]:
        return filter(lambda f: f.type == frame_type, self.frames)

    @property
    def stream_frames(self) -> Iterator[StreamFrame[T_DICT]]:
        return map(lambda f: StreamFrame(f), self.frames_of_type(frame_types.STREAM))

    def stream_frames_of_stream(self, stream_id: int) -> Iterator[StreamFrame[T_DICT]]:
        return filter(lambda f: f.stream_id == stream_id, self.stream_frames)
