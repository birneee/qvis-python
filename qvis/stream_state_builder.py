from typing import Optional

from qvis import frame_types
from qvis.frame import Frame
from qvis.frames.reset_stream_frame import ResetStreamFrame
from qvis.frames.stop_sending_frame import StopSendingFrame
from qvis.frames.stream_frame import StreamFrame
from qvis.perspective import Perspective
from qvis.ranges import Ranges


class SendStreamBuilder:
    stream_id: int
    first_stream_frame_time: Optional[float] = None
    latest_stream_frame_time: Optional[float] = None
    used_0rtt: bool = False
    stop_sending_time: Optional[float] = None
    reset_time: Optional[float] = None
    _complete_time: Optional[float] = None
    """TODO never set yet"""

    def __init__(self, stream_id: int):
        self.stream_id = stream_id

    def apply_sent_frame(self, frame: Frame):
        if frame.type == frame_types.STREAM:
            self._apply_stream_frame(StreamFrame(frame))
        elif frame.type == frame_types.RESET_STREAM:
            reset_stream_frame = ResetStreamFrame(frame)
            self._apply_reset_stream_frame(reset_stream_frame)
        elif frame.type == frame_types.STOP_SENDING:
            raise RuntimeError(f'stream state error: sent {frame.type} frame for sent stream')

    def apply_received_frame(self, frame: Frame):
        if frame.type == frame_types.STOP_SENDING:
            stop_sending_frame = StopSendingFrame(frame)
            self._apply_stop_sending_frame(stop_sending_frame)
        elif frame.type == frame_types.RESET_STREAM or frame.type == frame_types.STREAM:
            raise RuntimeError(f'stream state error: received {frame.type} frame for sent stream')

    def _apply_reset_stream_frame(self, frame: ResetStreamFrame):
        if self.reset_time is None and self._complete_time is None:
            self.reset_time = frame.time

    def _apply_stop_sending_frame(self, frame: StopSendingFrame):
        if self.stop_sending_time is None and self._complete_time is None:
            self.stop_sending_time = frame.time

    def _apply_stream_frame(self, frame: StreamFrame):
        if self.first_stream_frame_time is None:
            self.first_stream_frame_time = frame.time
        self.latest_stream_frame_time = frame.time
        if frame.packet.header.packet_type == '0RTT':
            self.used_0rtt = True


class ReceiveStreamBuilder:
    stream_id: int
    first_stream_frame_time: Optional[float] = None
    latest_stream_frame_time: Optional[float] = None
    ranges: Ranges = Ranges([])
    complete_time: Optional[float] = None
    fin_offset: Optional[int] = None
    stop_sending_time: Optional[float] = None
    reset_time: Optional[float] = None
    used_0rtt: bool = False

    def __init__(self, stream_id: int):
        self.stream_id = stream_id

    def apply_received_frame(self, frame: Frame):
        if frame.type == frame_types.STREAM:
            stream_frame = StreamFrame(frame)
            self._apply_stream_frame(stream_frame)
        elif frame.type == frame_types.RESET_STREAM:
            reset_stream_frame = ResetStreamFrame(frame)
            self._apply_reset_stream_frame(reset_stream_frame)
        elif frame.type == frame_types.STOP_SENDING:
            raise RuntimeError(f'stream state error: received {frame.type} frame for receive stream')

    def apply_sent_frame(self, frame: Frame):
        if frame.type == frame_types.STOP_SENDING:
            stop_sending_frame = StopSendingFrame(frame)
            self._apply_stop_sending_frame(stop_sending_frame)
        elif frame.type == frame_types.RESET_STREAM or frame.type == frame_types.STREAM:
            raise RuntimeError(f'stream state error: sent {frame.type} frame for receive stream')

    def _apply_stream_frame(self, frame: StreamFrame):
        if self.stream_id is None:
            self.stream_id = frame.stream_id
        if self.stream_id != frame.stream_id:
            raise RuntimeError('stream id mismatch')
        self.ranges.add(range(frame.offset, frame.offset + frame.length))
        if self.first_stream_frame_time is None:
            self.first_stream_frame_time = frame.time
        self.latest_stream_frame_time = frame.time
        if frame.packet.header.packet_type == '0RTT':
            self.used_0rtt = True
        if frame.fin:
            self.fin_offset = frame.offset + frame.length
        if self.fin_offset is not None \
                and self.complete_time is None \
                and not self.ranges.has_missing_ranges\
                and self.reset_time is None \
                and self.stop_sending_time is None:
            self.complete_time = frame.time

    def _apply_stop_sending_frame(self, frame: StopSendingFrame):
        if self.stop_sending_time is None and self.complete_time is None:
            self.stop_sending_time = frame.time

    def _apply_reset_stream_frame(self, frame: ResetStreamFrame):
        if self.reset_time is None and self.complete_time is None:
            self.reset_time = frame.time

    @property
    def complete(self) -> bool:
        return self.complete_time is not None


class BidiStreamStateBuilder:
    """builder can be used to reconstruct the stream state step by step"""
    perspective: Perspective
    receive: ReceiveStreamBuilder
    send: SendStreamBuilder

    def __init__(self, stream_id: int, perspective: Perspective):
        self.perspective = perspective
        self.receive = ReceiveStreamBuilder(stream_id)
        self.send = SendStreamBuilder(stream_id)

    @property
    def stream_id(self) -> int:
        return self.receive.stream_id

    def apply_received_frame(self, frame: Frame):
        if frame.type == frame_types.STREAM \
                or frame.type == frame_types.RESET_STREAM:
            self.receive.apply_received_frame(frame)
        elif frame.type == frame_types.STOP_SENDING:
            self.send.apply_received_frame(frame)
        else:
            raise RuntimeError('unexpected frame')

    def apply_sent_frame(self, frame: Frame):
        if frame.type == frame_types.STREAM \
                or frame.type == frame_types.RESET_STREAM:
            self.send.apply_sent_frame(frame)
        elif frame.type == frame_types.STOP_SENDING:
            self.receive.apply_sent_frame(frame)
        else:
            raise RuntimeError('unexpected frame')
