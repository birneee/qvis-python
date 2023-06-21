from qvis import event_names
from qvis.event import Event


class RecoveryPacketLostEvent():
    base: Event

    def __init__(self, base: Event):
        if base.name != event_names.RECOVERY_PACKET_LOST:
            raise Exception("invalid name")
        self.base = base

    @property
    def packet_number(self) -> int:
        return int(self.base.data['header']['packet_number'])

    @property
    def trigger(self) -> str:
        return self.base.data['trigger']
