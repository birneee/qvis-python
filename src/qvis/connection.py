from __future__ import annotations

import datetime
import gzip
import os
import math
import pathlib
import time
from datetime import timedelta, datetime
from pathlib import Path
from typing import Iterator, Optional, TextIO, BinaryIO

import pandas as pd
import simdjson
from typing_extensions import Self

from qvis import event_names, frame_types
from qvis.cache import Cache
from qvis.frames.ack_frame import AckFrame
from qvis.frames.max_stream_data_frame import MaxStreamDataFrame
from qvis.frames.stream_frame import StreamFrame
from qvis.property_memory_cache import PropertyMemoryCache
from qvis.event import Event, XseRecord
from qvis.events.recovery_packet_lost_event import RecoveryPacketLostEvent
from qvis.events.transport_path_updated_event import TransportPathUpdatedEvent
from qvis.frame import Frame
from qvis.frames.datagram_frame import DatagramFrame
from qvis.packet import Packet
from qvis.recovery import MetricsUpdated
from qvis.utils.time_utils import print_func_time

PacketOffset = int
AckOffset = int
LostOffset = int
PacketNumber = int


def read_qlog(filepath: str | pathlib.Path) -> Connection:
    start = time.time()
    if str(filepath).endswith('.gz'):
        file = gzip.open(filepath, "rb")
        conn = parse_qlog(file)
    else:
        file = open(filepath, "rb")
        conn = parse_qlog(file)
    print(f'loaded {filepath} in {time.time() - start}s')
    return conn


# TODO rename
def parse_qlog(reader: TextIO) -> Connection:
    conn = Connection(
        Cache(Path(reader.name)),
        reader,
    )
    return conn


class Connection:
    """A QLOG QUIC Connection"""
    file_reader: BinaryIO
    json_parser: simdjson.Parser
    __cache__: Cache
    __property_memory_cache__: PropertyMemoryCache
    shift_ms: float
    min_ms: float
    max_ms: float

    def __init__(self, cache: Cache, file_reader: BinaryIO, shift_ms: float = 0, min_ms: float = -math.inf, max_ms: float = math.inf):
        self.__cache__ = cache
        self.__property_memory_cache__ = PropertyMemoryCache()
        self.file_reader = file_reader
        self.json_parser = simdjson.Parser(late_reuse_check=True)  # requires a fork: https://github.com/birneee/pysimdjson/tree/late_reuse_check
        self.shift_ms = shift_ms
        self.min_ms = min_ms
        self.max_ms = max_ms

    def copy(self) -> Connection:
        c = Connection(
            cache=self.__cache__,
            file_reader=self.file_reader,
            shift_ms=self.shift_ms,
            min_ms=self.min_ms,
            max_ms=self.max_ms,
        )
        c.__property_memory_cache__ = self.__property_memory_cache__
        return c

    def cut(self, start: timedelta = None, end: timedelta = None) -> Connection:
        if start is not None:
            self.min_ms = start.total_seconds() * 1000
        if end is not None:
            self.max_ms = end.total_seconds() * 1000
        return self

    @property
    def qlog_info(self) -> dict:
        r = self.file_reader
        r.seek(0)
        line = r.readline()
        try:
            return simdjson.loads(line)
        except Exception:
            raise Exception(f'failed to parse: {line}')

    @property
    def odcid(self) -> str:
        return self.qlog_info['trace']['common_fields']['ODCID']

    @property
    def is_client(self) -> bool:
        return self.qlog_info['trace']['vantage_point']['type'] == 'client'

    def events_of_type(self, name: str) -> Iterator[Event[simdjson.Object]]:
        return filter(lambda e: e.name == name, self.events)

    def received_xse_records(self, stream_id: int) -> Iterator[XseRecord]:
        return filter(lambda x: x.stream_id == stream_id,
                      map(lambda e: XseRecord(e),
                          self.events_of_type(event_names.TRANSPORT_XSE_RECORD_RECEIVED)))

    @property
    def sent_packets(self) -> Iterator[Packet]:
        for offset in self.sent_packet_file_offsets:
            packet = Packet(self.event_from_file_offset(offset))
            yield packet

    @property
    def sent_packet_file_offsets(self) -> list[int]:
        offsets = []
        for offset in self.event_line_offsets:
            event = self.event_from_file_offset(offset)
            if event.name == event_names.TRANSPORT_PACKET_SENT:
                offsets.append(offset)
        return offsets

    @property
    def received_packet_file_offsets(self) -> list[int]:
        offsets = []
        for offset in self.event_line_offsets:
            event = self.event_from_file_offset(offset)
            if event.name == event_names.TRANSPORT_PACKET_RECEIVED:
                offsets.append(offset)
        return offsets

    @property
    def received_packets(self) -> Iterator[Packet]:
        for offset in self.received_packet_file_offsets:
            p = Packet(self.event_from_file_offset(offset))
            if self.min_ms > -math.inf and p.time < self.min_ms:
                continue
            if self.max_ms < math.inf and p.time > self.max_ms:
                continue
            yield p

    @property
    def received_packets_reversed(self) -> Iterator[Packet]:
        for offset in reversed(self.received_packet_file_offsets):
            p = Packet(self.event_from_file_offset(offset))
            if self.min_ms > -math.inf and p.time < self.min_ms:
                continue
            if self.max_ms < math.inf and p.time > self.max_ms:
                continue
            yield p

    @property
    def sent_packets_reversed(self) -> Iterator[Packet]:
        for offset in reversed(self.sent_packet_file_offsets):
            p = Packet(self.event_from_file_offset(offset))
            if self.min_ms > -math.inf and p.time < self.min_ms:
                continue
            if self.max_ms < math.inf and p.time > self.max_ms:
                continue
            yield p

    @property
    def sent_frames(self) -> Iterator[Frame]:
        for packet in self.sent_packets:
            for frame in packet.frames:
                yield frame

    def sent_frames_of_type(self, frame_type: str) -> Iterator[Frame]:
        return filter(lambda f: f.type == frame_type, self.sent_frames)

    @property
    def sent_stream_frames(self) -> Iterator[StreamFrame]:
        return map(lambda f: StreamFrame(f), self.sent_frames_of_type(frame_types.STREAM))

    def sent_stream_frames_of_stream(self, stream_id: int) -> Iterator[StreamFrame]:
        return filter(lambda f: f.stream_id == stream_id, self.sent_stream_frames)

    @property
    def received_frames(self) -> Iterator[Frame[simdjson.Object]]:
        for packet in self.received_packets:
            for frame in packet.frames:
                yield frame

    @property
    def received_frames_reversed(self) -> Iterator[Frame[simdjson.Object]]:
        for packet in self.received_packets_reversed:
            for frame in packet.frames:
                yield frame

    @property
    def sent_frames_reversed(self) -> Iterator[Frame[simdjson.Object]]:
        for packet in self.sent_packets_reversed:
            for frame in packet.frames:
                yield frame

    def received_frames_of_type(self, frame_type: str) -> Iterator[Frame[simdjson.Object]]:
        return filter(lambda f: f.type == frame_type, self.received_frames)

    def received_frames_of_type_reversed(self, frame_type: str) -> Iterator[Frame[simdjson.Object]]:
        return filter(lambda f: f.type == frame_type, self.received_frames_reversed)

    def sent_frames_of_type_reversed(self, frame_type: str) -> Iterator[Frame[simdjson.Object]]:
        return filter(lambda f: f.type == frame_type, self.sent_frames_reversed)

    @property
    def received_stream_frames(self) -> Iterator[StreamFrame]:
        return map(lambda f: StreamFrame(f), self.received_frames_of_type(frame_types.STREAM))

    @property
    def received_stream_frames_reversed(self) -> Iterator[StreamFrame]:
        return map(lambda f: StreamFrame(f), self.received_frames_of_type_reversed(frame_types.STREAM))

    @property
    def sent_stream_frames_reversed(self) -> Iterator[StreamFrame]:
        return map(lambda f: StreamFrame(f), self.sent_frames_of_type_reversed(frame_types.STREAM))

    def received_stream_frames_of_stream(self, stream_id: int) -> Iterator[StreamFrame]:
        return filter(lambda f: f.stream_id == stream_id, self.received_stream_frames)

    def received_stream_frames_of_stream_reversed(self, stream_id: int) -> Iterator[StreamFrame]:
        return filter(lambda f: f.stream_id == stream_id, self.received_stream_frames_reversed)

    def sent_stream_frames_of_stream_reversed(self, stream_id: int) -> Iterator[StreamFrame]:
        return filter(lambda f: f.stream_id == stream_id, self.sent_stream_frames_reversed)

    def stream_flow_limit_sum_updates(self) -> Iterator[tuple[float, int]]:
        """sum of all stream flow limits"""
        """time in ms, maximum in bytes"""
        yield 0, self.remote_initial_max_stream_data_bidi_remote or 0
        stream_limits: dict[int, int] = {}
        for frame in self.received_frames:
            match frame.type:
                case frame_types.MAX_STREAM_DATA:
                    max_stream_data_frame = MaxStreamDataFrame(frame)
                    stream_limits[max_stream_data_frame.stream_id] = max_stream_data_frame.maximum
                    yield frame.time, sum(stream_limits.values())

    def remote_stream_flow_limit_updates(self, stream_id: int) -> Iterator[tuple[float, int]]:
        """time in ms, maximum in bytes"""
        yield 0, self.remote_initial_max_stream_data_bidi_remote or 0
        for frame in self.received_frames:
            match frame.type:
                case frame_types.MAX_STREAM_DATA:
                    max_stream_data_frame = MaxStreamDataFrame(frame)
                    if max_stream_data_frame.stream_id == stream_id:
                        yield frame.time, max_stream_data_frame.maximum

    def local_stream_flow_limit_updates(self, stream_id: int) -> Iterator[tuple[float, int]]:
        """time in ms, maximum in bytes"""
        yield 0, self.local_initial_max_stream_data_bidi_local or 0
        for frame in self.sent_frames:
            match frame.type:
                case frame_types.MAX_STREAM_DATA:
                    max_stream_data_frame = MaxStreamDataFrame(frame)
                    if max_stream_data_frame.stream_id == stream_id:
                        yield frame.time, max_stream_data_frame.maximum

    def remote_connection_flow_limit_updates(self) -> Iterator[tuple[float, int]]:
        """time in ms, maximum in bytes"""
        for frame in self.received_frames:
            if frame.type == frame_types.MAX_DATA:
                yield frame.time, frame.inner['maximum']

    @property
    def restored_parameters(self) -> dict | None:
        for event in self.events:
            if event.name == event_names.TRANSPORT_PARAMETERS_RESTORED:
                return event.data
        return None

    @property
    def remote_parameters(self) -> dict | None:
        for event in self.events:
            if event.name == event_names.TRANSPORT_PARAMETERS_SET:
                data = event.data
                if data is not None:
                    owner = data.get('owner')
                    if owner == 'remote':
                        return data
        return None

    @property
    def remote_initial_max_data(self) -> Optional[int]:
        """in bytes"""
        restored_parameters = self.restored_parameters
        if restored_parameters is not None:
            return restored_parameters.get('initial_max_data')
        return self.remote_parameters.get('initial_max_data')

    @property
    def remote_initial_max_stream_data_bidi_remote(self) -> Optional[int]:
        """in bytes"""
        restored_parameters = self.restored_parameters
        if restored_parameters is not None:
            return restored_parameters.get('initial_max_stream_data_bidi_remote')
        return self.remote_parameters.get('initial_max_stream_data_bidi_remote')

    @property
    def local_parameters(self) -> dict | None:
        for event in self.events:
            if event.name == event_names.TRANSPORT_PARAMETERS_SET:
                data = event.data
                if data is not None:
                    owner = data.get('owner')
                    if owner == 'local':
                        return data
        return None

    @property
    def local_initial_max_data(self) -> Optional[int]:
        """in bytes"""
        return self.local_parameters.get('initial_max_data')

    @property
    def local_initial_max_stream_data_bidi_local(self) -> Optional[int]:
        """in bytes"""
        return self.local_parameters.get('initial_max_stream_data_bidi_local')

    @property
    def bytes_in_flight_updates(self) -> Iterator[tuple[float, int]]:
        """time in ms, bytes in flight"""
        for event in self.events_of_type(event_names.RECOVERY_METRICS_UPDATED):
            metrics_updated = MetricsUpdated(event)
            bytes_in_flight = metrics_updated.bytes_in_flight
            if bytes_in_flight is not None:
                yield event.time, bytes_in_flight

    @property
    def max_time(self) -> float:
        """time in ms"""
        return self.last_event.time

    @property
    def congestion_window_updates(self) -> Iterator[tuple[float, int]]:
        """time in ms, maximum in bytes"""
        for event in self.events_of_type(event_names.RECOVERY_METRICS_UPDATED):
            metrics_updated = MetricsUpdated(event)
            congestion_window = metrics_updated.congestion_window
            if congestion_window is not None:
                yield event.time, congestion_window

    @property
    def avg_rtt(self) -> timedelta:
        rtt_sum = 0
        update_count = 0
        for event in self.events_of_type(event_names.RECOVERY_METRICS_UPDATED):
            metrics_updated = MetricsUpdated(event)
            rtt = metrics_updated.latest_rtt
            if rtt is not None:
                update_count += 1
                rtt_sum += rtt
        return timedelta(milliseconds=rtt_sum / update_count)

    @property
    def max_rtt(self) -> timedelta:
        max = 0
        for event in self.events_of_type(event_names.RECOVERY_METRICS_UPDATED):
            metrics_updated = MetricsUpdated(event)
            rtt = metrics_updated.latest_rtt
            if rtt is not None and rtt > max:
                max = rtt
        return timedelta(milliseconds=max)

    @property
    def min_rtt(self) -> Optional[timedelta]:
        min = math.inf
        for event in self._fast_events_of_type(event_names.RECOVERY_METRICS_UPDATED):
            metrics_updated = MetricsUpdated(event)
            rtt = metrics_updated.min_rtt
            if rtt is not None and rtt > 0 and rtt < min:
                min = rtt
        if min == math.inf:
            return None
        return timedelta(milliseconds=min)

    @property
    def rtt_updates(self) -> Iterator[tuple[float, float]]:
        """time in ms, latest rtt in ms"""
        for event in self.events_of_type(event_names.RECOVERY_METRICS_UPDATED):
            metrics_updated = MetricsUpdated(event)
            latest_rtt = metrics_updated.latest_rtt
            if latest_rtt is not None:
                yield event.time, latest_rtt

    def time_to_first_byte(self, stream_id: int) -> float:
        """time to first byte in ms"""
        for frame in self.received_stream_frames_of_stream(stream_id):
            if frame.length > 0:
                return frame.time

    def highest_acked_stream_updates(self, stream_id) -> Iterator[AckFrame, StreamFrame]:
        """received acks"""
        for ack_frame in self.received_frames_of_type(frame_types.ACK):
            ack_frame = AckFrame(ack_frame)
            for packet_number in ack_frame.acked_packet_numbers.iterate_elements_reversed():
                packet = self.sent_packet_by_number(packet_number)
                try:
                    stream_frame = next(packet.stream_frames_of_stream(stream_id))
                    yield ack_frame, stream_frame
                    break
                except StopIteration:
                    continue

    def avg_stream_receive_rate(self, stream_id) -> float:
        """in bits per second"""
        """if XSE-QUIC is used, this is the raw decrypted data including the control information"""
        start_time = self.time_to_first_byte(stream_id) / 1000  # in seconds
        stop_time = self.max_time / 1000  # in seconds
        duration = stop_time - start_time
        latest_stream_frame = next(self.received_stream_frames_of_stream_reversed(stream_id))
        max_stream_data = (latest_stream_frame.offset + latest_stream_frame.length) * 8  # in bits
        return max_stream_data / duration

    def avg_raw_xse_stream_receive_rate(self, stream_id) -> float:
        """in bits per second"""
        """the full length of the XSE-QUIC records"""
        start_time = self.time_to_first_byte(stream_id) / 1000  # in seconds
        stop_time = self.max_time / 1000  # in seconds
        duration = stop_time - start_time
        max_stream_data = sum(map(lambda x: x.raw_length, self.received_xse_records(stream_id))) * 8  # in bits
        return max_stream_data / duration

    def avg_xse_stream_receive_rate(self, stream_id) -> float:
        """in bits per second"""
        """only the payload of the XSE-QUIC records"""
        start_time = self.time_to_first_byte(stream_id) / 1000  # in seconds
        stop_time = self.max_time / 1000  # in seconds
        duration = stop_time - start_time
        max_stream_data = sum(map(lambda x: x.data_length, self.received_xse_records(stream_id))) * 8  # in bits
        return max_stream_data / duration

    @property
    def path_update_file_offsets(self) -> list[int]:
        offsets = []
        for offset in self.event_line_offsets:
            event = self.event_from_file_offset(offset)
            if event.name == event_names.TRANSPORT_PATH_UPDATED:
                offsets.append(offset)
        return offsets

    @property
    def path_updates(self) -> Iterator[TransportPathUpdatedEvent]:
        for offset in self.path_update_file_offsets:
            yield TransportPathUpdatedEvent(self.event_from_file_offset(offset))

    def received_datagram_frames(self) -> Iterator[DatagramFrame[simdjson.Object]]:
        for received_frame in self.received_frames_of_type(frame_types.DATAGRAM):
            yield DatagramFrame(received_frame)

    def sent_datagram_frames(self) -> Iterator[DatagramFrame]:
        for sent_frame in self.sent_frames_of_type(frame_types.DATAGRAM):
            yield DatagramFrame(sent_frame)

    def readline_from_offset(self, file_offset: int) -> str:
        r = self.file_reader
        r.seek(file_offset)
        return r.readline()

    def event_from_line(self, line: bytes, file_offset: int) -> Event[simdjson.Object]:
        if hasattr(self, 'current_event'):
            del self.current_event.inner
        self.current_event = Event[simdjson.Object](self.json_parser.parse(line), self, file_offset)
        return self.current_event

    def event_from_file_offset(self, file_offset: int) -> Event[simdjson.Object]:
        """event is temporary iterator"""
        r = self.file_reader
        r.seek(file_offset)
        line = r.readline()
        try:
            return self.event_from_line(line, file_offset)
        except Exception:
            raise Exception(f'failed to parse json: {line}')

    def iterate_json_lines(self) -> Iterator[(int, str)]:
        """iterate over tuple of offset and line string"""
        r = self.file_reader
        r.seek(0)
        offset = r.tell()
        line = r.readline()
        while line:
            if line.startswith(b'{') and line.endswith(b'}\n'):  # if json
                yield offset, line
            offset = r.tell()
            line = r.readline()

    @property
    def event_line_offsets(self) -> list[int]:
        iter = self.iterate_json_lines()
        next(iter, None)  # skip header
        offsets = []
        for offset, _ in iter:
            offsets.append(offset)
        return offsets

    @property
    def events(self) -> Iterator[Event[simdjson.Object]]:
        if os.environ.get('NOCACHE') == '1':
            iter = self.iterate_json_lines()
            next(iter)  # skip header
            for offset, line in iter:
                yield self.event_from_line(line, offset)
        else:
            for offset in self.event_line_offsets:
                yield self.event_from_file_offset(offset)

    @property
    def first_event(self) -> Optional[Event[simdjson.Object]]:
        iter = self.iterate_json_lines()
        try:
            next(iter)  # skip header
            offset, line = next(iter)
            return self.event_from_line(line, offset)
        except StopIteration:
            return None

    def first_event_of_type(self, name: str) -> Optional[Event[simdjson.Object]]:
        event = next(self.events_of_type(name), None)
        if event is None:
            return None
        return event

    @property
    def last_event(self) -> Optional[Event]:
        last_offset = self.event_line_offsets[-1]
        return self.event_from_file_offset(last_offset)

    @property
    @print_func_time
    def get_ack_and_loss_file_offsets(self) -> tuple[dict[PacketOffset, AckOffset], dict[PacketOffset, LostOffset]]:
        acks: dict[PacketOffset, AckOffset] = {}
        pending: dict[PacketNumber, PacketOffset] = {}
        loss: dict[PacketOffset, LostOffset] = {}

        for event in self.events2:
            if event.name == event_names.TRANSPORT_PACKET_SENT:
                packet = Packet(event)
                pending[packet.packet_number] = packet.event.file_offset
            elif event.name == event_names.TRANSPORT_PACKET_RECEIVED:
                packet = Packet(event)
                for frame in packet.frames:
                    if frame.type == frame_types.ACK:
                        ack = AckFrame(frame)
                        for packet_number in list(pending):
                            if packet_number in ack.acked_packet_numbers:
                                packet_offset = pending[packet_number]
                                del pending[packet_number]
                                acks[packet_offset] = ack.packet.event.file_offset
            elif event.name == event_names.RECOVERY_PACKET_LOST:
                lost = RecoveryPacketLostEvent(event)
                if lost.packet_number in pending:
                    packet_offset = pending[lost.packet_number]
                    del pending[lost.packet_number]
                    loss[packet_offset] = lost.base.file_offset
        print(f'{len(acks)} acked received packets')
        print(f'{len(pending)} pending received packets')
        print(f'{len(loss)} loss received packets')
        return acks, loss

    def get_first_ack_of_packet(self, packet: Packet) -> AckFrame | None:
        acks, loss = self.get_ack_and_loss_file_offsets
        if packet.event.file_offset in acks:
            packet = Packet(self.event_from_file_offset(acks[packet.event.file_offset]))
            for frame in packet.frames_of_type(frame_types.ACK):
                ack_frame = AckFrame(frame)
                if packet.packet_number in ack_frame.acked_packet_numbers:
                    return ack_frame
        return None

    @property
    def reference_time(self) -> float:
        return self.qlog_info['trace']['common_fields']['reference_time']

    @property
    def reference_time_as_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.reference_time / 1000)

    def align_time_to(self, conn: Self):
        own_time = self.qlog_info['trace']['common_fields']['reference_time']
        other_time = conn.qlog_info['trace']['common_fields']['reference_time']
        self.shift_ms = own_time - other_time

    def set_zero_time(self, time: datetime):
        own_time = self.qlog_info['trace']['common_fields']['reference_time']
        other_time = time.timestamp() * 1000
        self.shift_ms = own_time - other_time

    @property
    def total_received_datagram_payload(self) -> int:
        return sum(
            map(lambda d: d.length,
                map(lambda f: DatagramFrame(f),
                    self.received_frames_of_type(frame_types.DATAGRAM))))

    @property
    def total_sent_datagram_payload(self) -> int:
        return sum(
            map(lambda d: d.length,
                map(lambda f: DatagramFrame(f),
                    self.sent_frames_of_type(frame_types.DATAGRAM))))

    def total_sent_stream_payload(self, stream_id: int) -> int:
        try:
            frame = next(self.sent_stream_frames_of_stream_reversed(stream_id))
            return frame.offset + frame.length
        except StopIteration:
            return 0

    def total_received_stream_payload(self, stream_id: int) -> int:
        try:
            frame = next(self.received_stream_frames_of_stream_reversed(stream_id))
            return frame.offset + frame.length
        except StopIteration:
            return 0

    @property
    def start_time(self) -> timedelta:
        return self.first_event.time_as_timedelta

    @property
    def end_time(self) -> timedelta:
        return self.last_event.time_as_timedelta

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def ingress_datagram_goodput(self) -> float:
        """in Mbps"""
        return self.total_received_datagram_payload * 8 / 1E6 / self.duration.total_seconds()

    @property
    def handshake_completed_time_as_timedelta(self) -> Optional[timedelta]:
        for event in self.events_of_type(event_names.SECURITY_KEY_UPDATED):
            key_type = event.data.get('key_type')
            if key_type is None:
                continue
            if key_type == 'server_1rtt_secret' or key_type == 'client_1rtt_secret':
                return event.time_as_timedelta
        return None

    @property
    def handshake_confirmed_time_as_timedelta(self) -> Optional[timedelta]:
        if self.is_client:
            frame = next(self.received_frames_of_type(frame_types.HANDSHAKE_DONE))
            if frame is None:
                return None
            return frame.time_as_timedelta
        else:
            self.handshake_completed_time_as_timedelta

    def ingress_datagram_goodput_chunked(self, chunk_delta: Optional[timedelta]) -> list[(timedelta, float)]:
        d = pd.DataFrame(
            [(f.time_as_timedelta, DatagramFrame(f).length) for f in self._fast_received_frames_of_type(frame_types.DATAGRAM)],
            columns=['time', 'length']
        )
        if chunk_delta is not None:
            d['group'] = d['time'].apply(lambda t: math.floor(t / chunk_delta))
            d = d.groupby('group').agg(
                min_time=('time', min),
                sum_length=('length', sum)
            )
            d = d.reset_index(drop=True)
        d = list(d.itertuples(index=False, name=None))
        return d

    def chunks(self, chunk_delta: timedelta) -> Iterator[Connection]:
        chunk_start = chunk_delta * math.floor(self.start_time / chunk_delta)
        chunk_end = chunk_start + chunk_delta
        while chunk_start < self.end_time:
            c = self.copy()
            c.cut(start=chunk_start, end=chunk_end)
            yield c
            chunk_start = chunk_end
            chunk_end += chunk_delta

    @property
    def alpn(self) -> str:
        """only valid for QUIC QLOGs."""
        event = self.first_event_of_type('transport:alpn_information')
        return event.data['chosen_alpn']
