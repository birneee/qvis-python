from __future__ import annotations

import json
from datetime import timedelta, datetime
import simdjson
from typing import Optional, Iterator, TYPE_CHECKING, TypeVar, Generic

if TYPE_CHECKING:
    from qvis.connection import Connection

T_DICT = TypeVar('T_DICT', simdjson.Object, dict)


class Event(Generic[T_DICT]):
    inner: T_DICT
    conn: Connection
    file_offset: int

    def __init__(self, inner: T_DICT, conn: Connection, file_offset):
        if 'time' not in inner:
            raise f'"{inner}" is not an event'
        self.inner = inner
        self.conn = conn
        self.file_offset = file_offset

    @property
    def index(self) -> int:
        """unique index of the event"""
        return getattr(self.inner, 'Index')

    @property
    def time_as_timedelta(self) -> timedelta:
        return timedelta(milliseconds=self.time)

    @property
    def time_as_datetime(self) -> datetime:
        return self.conn.reference_time_as_datetime + timedelta(milliseconds=self.time)

    @property
    def time(self) -> float:
        """in ms"""
        return self.inner['time'] + self.conn.shift_ms

    @property
    def name(self) -> str:
        """name of the event"""
        return self.inner['name']

    @property
    def data(self) -> Optional[T_DICT]:
        """event specific data"""
        return self.inner.get('data')

    def subsequent_events(self) -> Iterator[Event]:
        r = self.conn.file_reader
        r.seek(self.file_offset)
        r.readline()  # skip this element
        current_offset = r.tell()
        line = r.readline()
        while line:
            next_offset = r.tell()
            yield Event(json.loads(line), self.conn, current_offset)
            current_offset = next_offset
            r.seek(current_offset)
            line = r.readline()

    def subsequent_events_of_type(self, type_name: str) -> Iterator[Event[T_DICT]]:
        return filter(lambda e: e.name == type_name, self.subsequent_events())

    def as_dict(self) -> Event[dict]:
        """convert temporary Event[JSONObject] to copy Event[dict]"""
        if isinstance(self.inner, simdjson.Object):
            return Event(self.inner.as_dict(), self.conn, self.file_offset)
        else:
            return self.conn.event_from_file_offset(self.file_offset)


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
