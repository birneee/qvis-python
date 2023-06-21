from qvis.event import Event
from qvis.event_names import TRANSPORT_PATH_UPDATED
from typing import Optional


class TransportPathUpdatedEvent(Event):

    def __init__(self, base: Event):
        super().__init__(base.inner, base.conn, file_offset=base.file_offset)
        if base.name != TRANSPORT_PATH_UPDATED:
            raise Exception("invalid name")

    @property
    def dst_ip(self) -> Optional[str]:
        if self.data is None:
            return None
        return self.data.get('dst_ip')

    @property
    def dst_port(self) -> Optional[str]:
        if self.data is None:
            return None
        dst_port_str = self.data.get('dst_port')
        if dst_port_str is None:
            return None
        else:
            return int(dst_port_str)
