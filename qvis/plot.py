import itertools
import time
from typing import Iterator, Optional, TypeVar, Callable

import matplotlib.transforms as transforms
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from .connection import Connection


def byte_axis_formatter(bytes: int, position: int) -> str:
    if bytes > 1000000:
        return f'{bytes / 1000000:.0f}M'
    if bytes > 1000:
        return f'{bytes / 1000:.0f}K'
    return f'{bytes:.0f}'


QvisTimeAxisFormatter = FuncFormatter(lambda seconds, position: f'{seconds:.0f}')
QvisByteAxisFormatter = FuncFormatter(byte_axis_formatter)


def extend_time(conn: Connection, values: Iterator[tuple[float, any]]) -> Iterator[tuple[float, any]]:
    value = None
    for new_value in values:
        value = new_value
        yield value
    if value is not None:
        yield conn.max_time, value[1]


def increasing_only(values: Iterator[tuple[float, int]]) -> Iterator[tuple[float, int]]:
    """helper function"""
    """ignore non increasing values"""
    max_value = 0
    for time, value in values:
        if value > max_value:
            yield time, value
            max_value = value


def add_updates(u1: Iterator[tuple[float, int]], u2: Iterator[tuple[float, int]]) -> Iterator[tuple[float, int]]:
    """helper function"""
    """combine two update streams, by adding values"""
    return combine_updates(lambda a, b: a + b, u1, u2)


def subtract_updates(u1: Iterator[tuple[float, int]], u2: Iterator[tuple[float, int]]) -> Iterator[tuple[float, int]]:
    """helper function"""
    """combine two update streams, by subtracting values"""
    return combine_updates(lambda a, b: a - b, u1, u2)


def combine_updates(combine: Callable[[int, int], int], u1: Iterator[tuple[float, int]],
                    u2: Iterator[tuple[float, int]]) -> Iterator[tuple[float, int]]:
    """helper function"""
    """combine two update streams"""
    try:
        current_time1: float = 0
        current_value1: int = 0
        next_time1, next_value1 = next(u1)
        current_time2: float = 0
        current_value2: int = 0
        next_time2, next_value2 = next(u2)

        def update1():
            nonlocal current_time1, current_value1, next_time1, next_value1
            if next_time1 == float('inf'):
                raise StopIteration
            current_time1 = next_time1
            current_value1 = next_value1
            try:
                next_time1, next_value1 = next(u1)
            except StopIteration:
                next_time1 = float('inf')
                next_value1 = None

        def update2():
            nonlocal current_time2, current_value2, next_time2, next_value2
            if next_time2 == float('inf'):
                raise StopIteration
            current_time2 = next_time2
            current_value2 = next_value2
            try:
                next_time2, next_value2 = next(u2)
            except StopIteration:
                next_time2 = float('inf')
                next_value2 = None

        update1()
        update2()
        while True:
            if current_time2 > next_time1:
                update1()
                continue
            if current_time1 > next_time2:
                update2()
                continue
            if current_time1 > current_time2:
                yield current_time1, combine(current_value1, current_value2)
            else:
                yield current_time2, combine(current_value1, current_value2)
            if next_time1 > next_time2:
                update2()
            else:
                update1()
    except StopIteration:
        return


K = TypeVar("K")
V = TypeVar("V")


def unzip(it: Iterator[tuple[K, V]]) -> (tuple[K], tuple[V]):
    """helper function"""
    """unzip stream of tuples to a tuple of streams"""
    result = tuple(zip(*it))
    if len(result) == 0:
        return (), ()
    else:
        return result


def plot_remote_stream_flow_limit(ax: Axes, conn: Connection, stream_id: int, color: str = '#ff69b4',
                                  label: str | None = 'Stream flow control limits', linestyle: str = 'solid'):
    start = time.time()
    updates = conn.remote_stream_flow_limit_updates(stream_id)
    updates = extend_time(conn, updates)
    updates = increasing_only(updates)
    ms, values = unzip(updates)
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=values, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)
    print(f'plotted in {time.time() - start}s')


def plot_local_stream_flow_limit(ax: Axes, conn: Connection, stream_id: int, color: str = '#ff69b4',
                                 label: str | None = 'Stream flow control limits', linestyle: str = 'solid'):
    start = time.time()
    updates = conn.local_stream_flow_limit_updates(stream_id)
    updates = extend_time(conn, updates)
    updates = increasing_only(updates)
    ms, limits = unzip(updates)
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=limits, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)
    print(f'plotted in {time.time() - start}s')


def plot_remote_connection_flow_limit(ax: Axes, conn: Connection, color: str = '#a80f3a',
                                      label: str | None = 'Connection flow control limit', linestyle: str = 'solid'):
    start = time.time()
    updates = conn.remote_connection_flow_limit_updates()
    updates = extend_time(conn, updates)
    updates = increasing_only(updates)
    ms, limits = unzip(updates)
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=limits, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)
    print(f'plotted in {time.time() - start}s')


def plot_congestion_window(ax: Axes, conn: Connection, color: str = '#8a2be2', label: str | None = 'Congestion window',
                           linestyle: str = 'solid'):
    start = time.time()
    ms, window = zip(*extend_time(conn, conn.congestion_window_updates))
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=window, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)
    print(f'plotted in {time.time() - start}s')


def plot_available_congestion_window_of_stream(ax: Axes, conn: Connection, stream_id: int, color: str = '#8a2be2',
                                               label: str | None = 'Congestion window',
                                               linestyle: str = 'solid'):
    start = time.time()
    stream = increasing_only(map(lambda s: (s.time, s.offset + s.length), conn.sent_stream_frames_of_stream(stream_id)))
    in_flight = conn.bytes_in_flight_updates
    congestion = extend_time(conn, conn.congestion_window_updates)
    available = add_updates(stream, subtract_updates(congestion, in_flight))
    ms, value = unzip(available)
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=value, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)
    print(f'plotted in {time.time() - start}s')


def plot_rtt(ax: Axes, conn: Connection, color: str = '#ff9900', label: str | None = 'Latest RTT',
             linestyle: str = 'solid'):
    start = time.time()
    ms, window = zip(*extend_time(conn, conn.rtt_updates))
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=window, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)
    print(f'plotted in {time.time() - start}s')


def plot_bytes_in_flight(ax: Axes, conn: Connection, color: str = '#808000', label: str | None = 'Bytes in flight'):
    start = time.time()
    ms, in_flight = zip(*conn.bytes_in_flight_updates)
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=in_flight, edges=seconds, baseline=None, color=color, label=label)
    print(f'plotted in {time.time() - start}s')


def plot_raw_data_sent(ax: Axes, conn: Connection, color: str = '#0000ff',
                       label: str = 'Data sent (includes retransmits)'):
    start = time.time()
    ms, length = zip(*map(lambda f: (f.time, f.raw_length),
                          conn.sent_packets))
    seconds = list(map(lambda m: m / 1000, ms))
    cum_length = np.cumsum(length)
    ax.scatter(x=seconds, y=cum_length, s=1.5, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


def plot_stream_data_sent(ax: Axes, conn: Connection, stream_id: int, color: str = '#0000ff',
                          label: str = 'Stream data sent'):
    start = time.time()
    ms, cum_length = zip(*map(lambda f: (f.time, f.offset + f.length),
                              conn.sent_stream_frames_of_stream(stream_id)))
    seconds = list(map(lambda m: m / 1000, ms))
    ax.scatter(x=seconds, y=cum_length, s=1.5, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


def plot_received_acks_of_stream(ax: Axes, conn: Connection, stream_id: int, color: str = '#6b8e23',
                                 label: str | None = 'Data acknowledged'):
    start = time.time()
    ms, cum_length = zip(*map(lambda f: (f[0].time, f[1].offset + f[1].length),
                              conn.highest_acked_stream_updates(stream_id)))
    seconds = list(map(lambda m: m / 1000, ms))
    ax.scatter(x=seconds, y=cum_length, s=1.5, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


def plot_stream_data_received(ax: Axes, conn: Connection, stream_id: int, color: str = '#0000ff',
                              label: str = 'Stream data received'):
    start = time.time()
    ms, cum_length = zip(*map(lambda f: (f.time, f.offset + f.length),
                              conn.received_stream_frames_of_stream(stream_id)))
    seconds = list(map(lambda m: m / 1000, ms))
    ax.scatter(x=seconds, y=cum_length, s=1.5, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


def plot_xse_data_received(ax: Axes, conn: Connection, stream_id: int, color: str = '#ff00c7',
                           label: str = 'XSE data received'):
    start = time.time()
    ms, lengths = unzip(map(lambda x: (x.time, x.data_length),
                        conn.received_xse_records(stream_id)))
    seconds = list(map(lambda m: m / 1000, ms))
    cum_lengths = np.cumsum(lengths)
    ax.scatter(x=seconds, y=cum_lengths, s=1.5, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


def plot_time_to_first_byte(ax: Axes, conn: Connection, stream_id: int, color: str = 'black',
                            label: Optional[str] = 'Time to first byte'):
    ttfb = conn.time_to_first_byte(stream_id)
    ax.scatter(ttfb / 1000, 0, marker='^', color=color, label=label, clip_on=False, zorder=100, linewidth=1, s=12,
               # path_effects=[path_effects.SimpleLineShadow(shadow_color='red', offset=(0.5, 0.5)), path_effects.Normal()],
               transform=transforms.offset_copy(ax.transData, fig=ax.figure, x=0, y=-2.5, units='points')
               )
