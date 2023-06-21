from __future__ import annotations

from datetime import timedelta, datetime
from typing import Optional, Iterable, Iterator, TYPE_CHECKING

from cysimdjson import JSONObject

from qvis import event_names
from qvis.event import Event

if TYPE_CHECKING:
    from qvis.connection import Connection


class EventIterator:
    inner: JSONObject
    conn: Connection
    file_offset: int

    def __init__(self, inner: JSONObject, conn: Connection, file_offset):
        self.inner = inner
        self.conn = conn
        self.file_offset = file_offset

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
    def data(self) -> Optional[JSONObject]:
        """event specific data"""
        return self.inner.get('data')

    def to_event(self) -> Event:
        return self.conn.event_from_file_offset(self.file_offset)