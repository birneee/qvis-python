from qvis import event_names, frame_types
from qvis.event import Event
from qvis.frame import Frame
from qvis.packet import Packet
from qvis.perspective import Perspective
from qvis.stream_state_builder import BidiStreamStateBuilder


class ConnectionStateBuilder:
    """builder can be used to reconstruct the quic connection state step by step"""
    bidi_stream_states: dict[int, BidiStreamStateBuilder]
    perspective: Perspective

    def __init__(self, perspective: Perspective):
        self.perspective = perspective
        self.bidi_stream_states = {}

    def apply_event(self, event: Event):
        if event.name == event_names.TRANSPORT_PACKET_SENT:
            self._apply_sent_packet(Packet(event))
        elif event.name == event_names.TRANSPORT_PACKET_RECEIVED:
            self._apply_received_packet(Packet(event))

    def _apply_sent_packet(self, packet: Packet):
        for frame in packet.frames:
            self._apply_sent_frame(frame)

    def _apply_received_packet(self, packet: Packet):
        for frame in packet.frames:
            self._apply_received_frame(frame)

    def _apply_sent_frame(self, frame: Frame):
        if frame.type == frame_types.STREAM \
                or frame.type == frame_types.STOP_SENDING \
                or frame.type == frame_types.RESET_STREAM:
            self._get_bidi_stream_state_builder(frame['stream_id']).apply_sent_frame(frame)

    def _apply_received_frame(self, frame: Frame):
        if frame.type == frame_types.STREAM \
                or frame.type == frame_types.STOP_SENDING \
                or frame.type == frame_types.RESET_STREAM:
            self._get_bidi_stream_state_builder(frame['stream_id']).apply_received_frame(frame)

    def _get_bidi_stream_state_builder(self, stream_id: int) -> BidiStreamStateBuilder:
        if stream_id not in self.bidi_stream_states:
            ssb = BidiStreamStateBuilder(stream_id, self.perspective)
            self.bidi_stream_states[stream_id] = ssb
        return self.bidi_stream_states[stream_id]
