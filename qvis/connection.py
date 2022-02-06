from __future__ import annotations

import gzip
import math
import time
from typing import Iterator, Optional, TextIO

import ujson as json

from . import event_names, frame_types
from .event import Event, XseRecord
from .frame import Frame, MaxStreamDataFrame, StreamFrame, AckFrame
from .packet import Packet
from .recovery import MetricsUpdated


def read_qlog(filepath: str, shift_ms: float = 0, max_ms: float = math.inf) -> Connection:
    start = time.time()
    if filepath.endswith('.gz'):
        with gzip.open(filepath) as file:
            conn = parse_qlog(file, shift_ms, max_ms)
    else:
        with open(filepath) as file:
            conn = parse_qlog(file, shift_ms, max_ms)
    print(f'loaded {filepath} in {time.time() - start}s')
    return conn


def parse_qlog(reader: TextIO, shift_ms: float = 0, max_ms: float = math.inf) -> Connection:
    qlog_info = json.loads(next(reader))
    raw_events = []
    sent_packet_events: list[dict] = []
    sent_packet_numbers: dict[int, int] = {}  # packet_number, index in events
    received_packet_events: list[dict] = []
    received_packet_numbers: dict[int, int] = {}  # packet_number, index in events
    for index, line in enumerate(reader):
        event = json.loads(line)
        if shift_ms != 0:
            event['time'] += shift_ms
        if event['time'] > max_ms:
            break
        raw_events.append(event)
        match event['name']:
            case event_names.TRANSPORT_PACKET_SENT:
                sent_packet_numbers[event['data']['header']['packet_number']] = index
                sent_packet_events.append(event)
            case event_names.TRANSPORT_PACKET_RECEIVED:
                packet_number: Optional[int] = event['data']['header'].get('packet_number')
                if packet_number is not None:  # because retry packets do not have a packet_number
                    received_packet_numbers[packet_number] = index
                    received_packet_events.append(event)
    conn = Connection(
        qlog_info,
        raw_events,
        sent_packet_numbers,
        received_packet_numbers,
        received_packet_events,
        sent_packet_events
    )
    return conn


class Connection:
    """A QLOG QUIC Connection"""
    qlog_info: dict
    raw_events: list[dict]
    sent_packet_events: list[dict] = []
    sent_packet_numbers: dict[int, int] = {}  # packet_number, index in events
    received_packet_events: list[dict] = []
    received_packet_numbers: dict[int, int] = {}  # packet_number, index in events

    def __init__(self, qlog_info: dict, raw_events: list[dict], sent_packet_numbers: dict[int, int],
                 received_packet_numbers: dict[int, int], received_packet_events: list[dict],
                 sent_packet_events: list[dict]):
        self.qlog_info: qlog_info
        self.raw_events = raw_events
        self.sent_packet_numbers = sent_packet_numbers
        self.received_packet_numbers = received_packet_numbers
        self.received_packet_events = received_packet_events
        self.sent_packet_events = sent_packet_events

    @property
    def events(self) -> Iterator[Event]:
        for raw_event in self.raw_events:
            yield Event(raw_event)

    def events_of_type(self, name: str) -> Iterator[Event]:
        return filter(lambda e: e.name == name, self.events)

    def received_xse_records(self, stream_id: int) -> Iterator[XseRecord]:
        return filter(lambda x: x.stream_id == stream_id,
                      map(lambda e: XseRecord(e),
                          self.events_of_type(event_names.TRANSPORT_XSE_RECORD_RECEIVED)))

    @property
    def sent_packets(self) -> Iterator[Packet]:
        for raw_event in self.sent_packet_events:
            yield Packet(Event(raw_event))

    @property
    def received_packets(self) -> Iterator[Packet]:
        for raw_event in self.received_packet_events:
            yield Packet(Event(raw_event))

    @property
    def received_packets_reversed(self) -> Iterator[Packet]:
        for raw_event in reversed(self.received_packet_events):
            yield Packet(Event(raw_event))

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
    def received_frames(self) -> Iterator[Frame]:
        for packet in self.received_packets:
            for frame in packet.frames:
                yield frame

    @property
    def received_frames_reversed(self) -> Iterator[Frame]:
        for packet in self.received_packets_reversed:
            for frame in packet.frames:
                yield frame

    def received_frames_of_type(self, frame_type: str) -> Iterator[Frame]:
        return filter(lambda f: f.type == frame_type, self.received_frames)

    def received_frames_of_type_reversed(self, frame_type: str) -> Iterator[Frame]:
        return filter(lambda f: f.type == frame_type, self.received_frames_reversed)

    @property
    def received_stream_frames(self) -> Iterator[StreamFrame]:
        return map(lambda f: StreamFrame(f), self.received_frames_of_type(frame_types.STREAM))

    @property
    def received_stream_frames_reversed(self) -> Iterator[StreamFrame]:
        return map(lambda f: StreamFrame(f), self.received_frames_of_type_reversed(frame_types.STREAM))

    def received_stream_frames_of_stream(self, stream_id: int) -> Iterator[StreamFrame]:
        return filter(lambda f: f.stream_id == stream_id, self.received_stream_frames)

    def received_stream_frames_of_stream_reversed(self, stream_id: int) -> Iterator[StreamFrame]:
        return filter(lambda f: f.stream_id == stream_id, self.received_stream_frames_reversed)

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
        return self.raw_events[-1]["time"]

    @property
    def congestion_window_updates(self) -> Iterator[tuple[float, int]]:
        """time in ms, maximum in bytes"""
        for event in self.events_of_type(event_names.RECOVERY_METRICS_UPDATED):
            metrics_updated = MetricsUpdated(event)
            congestion_window = metrics_updated.congestion_window
            if congestion_window is not None:
                yield event.time, congestion_window

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

    def sent_packet_by_number(self, packet_number: int) -> Packet:
        return Packet(Event(self.raw_events[self.sent_packet_numbers[packet_number]]))

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

    def avg_xse_stream_receive_rate(self, stream_id) -> float:
        """in bits per second"""
        """only the payload of the XSE-QUIC records"""
        start_time = self.time_to_first_byte(stream_id) / 1000  # in seconds
        stop_time = self.max_time / 1000  # in seconds
        duration = stop_time - start_time
        max_stream_data = sum(map(lambda x: x.data_length, self.received_xse_records(stream_id))) * 8  # in bits
        return max_stream_data / duration
