from typing import Optional

from .event import Event


class MetricsUpdated:
    event: Event

    def __init__(self, event: Event):
        self.event = event

    @property
    def congestion_window(self) -> Optional[int]:
        return self.event.data.get('congestion_window')

    @property
    def bytes_in_flight(self) -> Optional[int]:
        return self.event.data.get('bytes_in_flight')

    @property
    def latest_rtt(self) -> Optional[float]:
        """returns rtt in ms"""
        return self.event.data.get('latest_rtt')
