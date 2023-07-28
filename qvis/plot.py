import statistics
import time
from typing import Iterator, Optional, TypeVar, Callable, Iterable

import matplotlib.transforms as transforms
import numpy as np
import matplotlib as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

from .connection import Connection
from .frames.datagram_frame import DatagramFrame
from .frames.stream_frame import StreamFrame
from .utils.iterator import window
from datetime import timedelta

from .utils.time import print_func_time


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
    max_value = -1
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
T = TypeVar("T")


def unzip(it: Iterable[tuple[K, V]]) -> (tuple[K], tuple[V]):
    """helper function"""
    """unzip stream of tuples to a tuple of streams"""
    result = tuple(zip(*it))
    if len(result) == 0:
        return (), ()
    else:
        return result


def unzip3(it: Iterable[tuple[K, V, T]]) -> (tuple[K], tuple[V], tuple[T]):
    """helper function"""
    """unzip stream of tuples to a tuple of streams"""
    result = tuple(zip(*it))
    if len(result) == 0:
        return (), (), ()
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
                           linestyle: str = 'solid', linewidth: float = 1):
    start = time.time()
    ms, window = zip(*extend_time(conn, conn.congestion_window_updates))
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=window, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle,
              linewidth=linewidth)
    print(f'plotted in {time.time() - start}s')

@print_func_time
def plot_available_congestion_window_on_datagram(ax: Axes, conn: Connection, color: str = '#8a2be2',
                                               label: str | None = 'Congestion window',
                                               linestyle: str = 'solid', start_offset: int = 0, hide_if_empty: bool = True):
    offset = start_offset

    def extract(d: DatagramFrame) -> (float, int):
        nonlocal offset
        offset += d.length
        return d.time_as_timedelta.total_seconds(), offset
    sent_datagrams = conn.sent_datagram_frames()
    updated = list(map(extract, sent_datagrams))
    if hide_if_empty and len(updated) == 0:
        return

    plot_available_congestion_window(ax, conn, color=color, label=label, linestyle=linestyle, y_zero=updated)

def plot_available_congestion_window(ax: Axes, conn: Connection, color: str = '#8a2be2',
                                               label: str | None = 'Congestion window',
                                               linestyle: str = 'solid', y_zero: Iterator[tuple[float, int]] = [(0,0)]):
    stream = increasing_only(y_zero)
    in_flight = map(lambda u: (u[0] / 1000, u[1]), conn.bytes_in_flight_updates)
    congestion = map(lambda u: (u[0] / 1000, u[1]), conn.congestion_window_updates)
    available = add_updates(stream, subtract_updates(congestion, in_flight))
    seconds, value = unzip(available)
    seconds = list(seconds)
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=value, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)

def plot_available_congestion_window_of_stream(ax: Axes, conn: Connection, stream_id: int, color: str = '#8a2be2',
                                               label: str | None = 'Congestion window',
                                               linestyle: str = 'solid', chunk_size: int = 1):
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    start = time.time()
    stream = increasing_only(map(lambda s: (s.time, s.offset + s.length), conn.sent_stream_frames_of_stream(stream_id)))
    in_flight = conn.bytes_in_flight_updates
    congestion = extend_time(conn, conn.congestion_window_updates)
    available = add_updates(stream, subtract_updates(congestion, in_flight))
    ms, value = unzip(available)
    seconds = list(map(lambda m: m / 1000, ms))
    if chunk_size != 1:
        value = list(map(lambda v: statistics.mean(v), chunker(value, chunk_size)))
        seconds = list(map(lambda v: statistics.mean(v), chunker(seconds, chunk_size)))
    seconds.append(conn.max_time / 1000)
    ax.stairs(values=value, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle)
    print(f'plotted in {time.time() - start}s')


def plot_rtt(ax: Axes, conn: Connection, color: str = '#ff9900', label: str | None = 'Latest RTT',
             linestyle: str = 'solid', rtt_ms_step_size: float = 1, linewidth: float = 1):
    start = time.time()
    ms, updates = zip(*extend_time(conn, conn.rtt_updates))
    seconds = list(map(lambda m: m / 1000, ms))
    seconds.append(conn.max_time / 1000)
    if rtt_ms_step_size != 0:
        updates = list(map(lambda u: rtt_ms_step_size * round(u / rtt_ms_step_size), updates))
    ax.stairs(values=updates, edges=seconds, baseline=None, color=color, label=label, linestyle=linestyle,
              linewidth=linewidth)
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


@print_func_time
def plot_stream_data_sent(ax: Axes, conn: Connection, stream_id: int, color: str = '#0000ff',
                          label: str = 'Sent stream payload', linewidth: float = 4, hide_if_empty: bool = True, y_offset: int = 0):
    seconds, start_offsets, end_offsets = unzip3(map(lambda f: (f.time_as_timedelta.total_seconds(), y_offset + f.offset, y_offset + f.offset + f.length),
             conn.sent_stream_frames_of_stream(stream_id)))
    if hide_if_empty and len(seconds) == 0:
        return
    ax.vlines(x=seconds, ymin=start_offsets, ymax=end_offsets, rasterized=True, color=color, label=label,
              linewidth=linewidth)


def plot_received_acks_of_stream(ax: Axes, conn: Connection, stream_id: int, color: str = '#6b8e23',
                                 label: str | None = 'Data acknowledged'):
    start = time.time()
    ms, cum_length = zip(*map(lambda f: (f[0].time, f[1].offset + f[1].length),
                              conn.highest_acked_stream_updates(stream_id)))
    seconds = list(map(lambda m: m / 1000, ms))
    ax.scatter(x=seconds, y=cum_length, s=1.5, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


@print_func_time
def plot_received_ack_delay(ax: Axes, conn: Connection, stream_id: int):
    """TODO WIP"""
    def add_ack(sf: StreamFrame) -> (StreamFrame, AckFrame | None):
        return sf, conn.get_first_ack_of_packet(sf.packet)

    def to_plot_values(sf: StreamFrame, af: AckFrame) -> (float, float, float):
        return sf.offset + sf.length / 2, sf.time_as_timedelta.total_seconds(), af.time_as_timedelta.total_seconds()

    offsets, start_seconds, end_seconds = unzip3(
        map(lambda t: to_plot_values(t[0], t[1]),
            filter(lambda t: t[1] != None,
                   map(add_ack,
                       conn.sent_stream_frames_of_stream(stream_id)))))
    ax.hlines(y=offsets, xmin=start_seconds, xmax=end_seconds)


@print_func_time
def plot_stream_data_received(ax: Axes, conn: Connection, stream_id: int, color: str = '#0000ff',
                              label: str = 'Received stream payload', linewidth: float = 4, hide_if_empty: bool = True, y_offset: int = 0):
    seconds, start_offsets, end_offsets = unzip3(map(lambda f: (f.time_as_timedelta.total_seconds(), y_offset + f.offset, y_offset + f.offset + f.length),
             conn.received_stream_frames_of_stream(stream_id)))
    if hide_if_empty and len(seconds) == 0:
        return
    ax.vlines(x=seconds, ymin=start_offsets, ymax=end_offsets, rasterized=True, color=color, label=label,
              linewidth=linewidth)


def plot_xse_data_received(ax: Axes, conn: Connection, stream_id: int, color: str = '#ff00c7',
                           label: str = 'XSE data received'):
    start = time.time()
    ms, lengths = unzip(map(lambda x: (x.time, x.data_length),
                            conn.received_xse_records(stream_id)))
    seconds = list(map(lambda m: m / 1000, ms))
    cum_lengths = np.cumsum(lengths)
    ax.scatter(x=seconds, y=cum_lengths, s=1.5, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


def plot_received_xse_overhead_ratio(ax: Axes, conn: Connection, stream_id: int, color: str = '#ff00c7',
                                     label: str = 'XSE overhead ratio', shift_ms: float = 0, markersize: float = 1.5):
    start = time.time()
    ms, ratio = unzip(map(lambda x: (x.time, (x.raw_length - x.data_length) / x.data_length),
                          conn.received_xse_records(stream_id)))
    seconds = list(map(lambda m: (m + shift_ms) / 1000, ms))
    ax.scatter(x=seconds, y=list(ratio), s=markersize, rasterized=True, label=label, color=color)
    print(f'plotted in {time.time() - start}s')


def plot_time_to_first_byte(ax: Axes, conn: Connection, stream_id: int, color: str = 'black',
                            label: Optional[str] = 'Time to first byte'):
    ttfb = conn.time_to_first_byte(stream_id)
    ax.scatter(ttfb / 1000, 0, marker='^', color=color, label=label, clip_on=False, zorder=100, linewidth=1, s=12,
               # path_effects=[path_effects.SimpleLineShadow(shadow_color='red', offset=(0.5, 0.5)), path_effects.Normal()],
               transform=transforms.offset_copy(ax.transData, fig=ax.figure, x=0, y=-2.5, units='points')
               )


@print_func_time
def plot_path_updates(ax: Axes, conn: Connection, label: str = "Path updates", linestyle: str | None = 'dashed',
                      color: str | None = "black"):
    path_updates = conn.path_updates
    for i, path_update in enumerate(path_updates):
        ax.axvline(path_update.time_as_timedelta.total_seconds(), label=label if i == 0 else None, linestyle=linestyle, color=color)
        ax.text(path_update.time_as_timedelta.total_seconds(), 1.01, f'{path_update.dst_ip}:{path_update.dst_port}',
                transform=ax.get_xaxis_text1_transform(0)[0], ha='center')


@print_func_time
def plot_vline_with_label(ax: Axes, time: timedelta, label: str, linestyle: str | None = 'dashed',
                      color: str | None = "black", y: float = 1.01, ha: str = 'left', va: str = 'top', rotation: float = 70, label_prefix: str = ' '):
    transform = plt.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(time.total_seconds(), ymin=0, ymax=y, color=color, transform=transform, clip_on=False, linestyle=linestyle)
    ax.text(time.total_seconds(), y, ' ' + label,
            transform=transform, ha=ha, va=va, rotation=rotation)


@print_func_time
def plot_vlines_with_labels(ax: Axes, xs: list[float], labels: list[str], linestyle: Optional[str] = 'dashed', color: Optional[str] = 'black', y_step: float = 0, y: float = 1, ha: str = 'left', rotation: float = 0, label_prefix: str = ' '):
    xs_with_labels = list(zip(xs, labels))
    if ha == 'left':
        xs_with_labels.sort(key=lambda e: e[0], reverse=True)
    for i, (x, label) in enumerate(xs_with_labels):
        plot_vline_with_label(ax, x, label, y=y+i*y_step, linestyle=linestyle, color=color, ha=ha, rotation=rotation, label_prefix=label_prefix)


@print_func_time
def plot_datagram_data_received(ax: Axes, conn: Connection, color: str | None = None,
                                label: str = 'Received datagram payload', linewidth: float = 4, start_offset: int = 0, hide_if_empty: bool = True):
    offset = start_offset

    def extract(d: DatagramFrame) -> (float, int, int):
        nonlocal offset
        offset += d.length
        return d.time_as_timedelta.total_seconds(), offset - d.length, offset

    received_datagrams = conn.received_datagram_frames()
    seconds, start_offsets, end_offsets = unzip3(map(extract, received_datagrams))
    if hide_if_empty and len(seconds) == 0:
        return
    ax.vlines(x=seconds, ymin=start_offsets, ymax=end_offsets, rasterized=True, color=color, label=label,
              linewidth=linewidth)


@print_func_time
def plot_datagram_data_sent(ax: Axes, conn: Connection, color: str | None = None,
                            label: str = 'Sent datagram payload', linewidth: float = 4, start_offset: int = 0, hide_if_empty: bool = True):
    offset = start_offset

    def extract(d: DatagramFrame) -> (float, int, int):
        nonlocal offset
        offset += d.length
        return d.time_as_timedelta.total_seconds(), offset - d.length, offset

    sent_datagrams = conn.sent_datagram_frames()
    seconds, start_offsets, end_offsets = unzip3(map(extract, sent_datagrams))
    if hide_if_empty and len(seconds) == 0:
        return
    ax.vlines(x=seconds, ymin=start_offsets, ymax=end_offsets, rasterized=True, color=color, label=label,
              linewidth=linewidth)


@print_func_time
def plot_stream_receive_gaps(ax: Axes, conn: Connection, stream_id: int, label: str | None = "Gap in received stream",
                             *args, **kwargs):
    plot_stream_gaps(ax, conn.received_stream_frames_of_stream(stream_id), label=label, *args, **kwargs)


@print_func_time
def plot_stream_send_gaps(ax: Axes, conn: Connection, stream_id: int, label: str | None = "Gap in sent stream", *args,
                          **kwargs):
    plot_stream_gaps(ax, conn.sent_stream_frames_of_stream(stream_id), label=label, *args, **kwargs)


def plot_stream_gaps(ax: Axes, stream_frames: Iterable[StreamFrame], min_time: timedelta = timedelta(0),
                     color: str | None = 'black', label: str = 'Gap in received stream', linewidth: float = 1):
    def has_min_time(a: StreamFrame, b: StreamFrame) -> bool:
        return b.time_as_timedelta - a.time_as_timedelta > min_time

    # def get_values(a: StreamFrame, b: StreamFrame) -> tuple[float, float, float]:
    #     return a.offset, a.time_as_timedelta.total_seconds(), b.time_as_timedelta.total_seconds()

    def get_values(a: StreamFrame, b: StreamFrame) -> tuple[float, float, float]:
        error = (b.time_as_timedelta.total_seconds() - a.time_as_timedelta.total_seconds()) / 2
        return a.time_as_timedelta.total_seconds() + error, a.offset, error

    seconds, offsets, errors = unzip3(
        map(lambda t: get_values(*t),
            filter(lambda t: has_min_time(*t),
                   window(
                       stream_frames))))

    ax.errorbar(seconds, offsets, xerr=errors, capsize=4, elinewidth=linewidth, color=color, label=label, fmt='none')

    for (x, y, error) in zip(seconds, offsets, errors):
        ms = error * 2 * 1000
        ax.annotate(f'{ms:.2f} ms', xy=(x, y), xytext=(0, -4), textcoords='offset pixels', ha='center', va='top',
                    color=color)


@print_func_time
def plot_datagram_receive_gaps(ax: Axes, conn: Connection, label: str = 'Gap in received datagrams', *args, **kwargs):
    plot_datagram_gaps(ax, conn.received_datagram_frames(), label=label, *args, **kwargs)


@print_func_time
def plot_datagram_send_gaps(ax: Axes, conn: Connection, label: str = 'Gap in sent datagrams', *args, **kwargs):
    plot_datagram_gaps(ax, conn.sent_datagram_frames(), label=label, *args, **kwargs)


def plot_datagram_gaps(ax: Axes, datagrams: Iterable[DatagramFrame], min_time: timedelta = timedelta(0),
                       color: str | None = 'black', label: str | None = None, linewidth: float = 1):
    def add_offset(datagrams: Iterable[DatagramFrame]) -> Iterator[tuple[int, DatagramFrame]]:
        offset = 0
        for datagram in datagrams:
            yield offset, datagram
            offset += datagram.length

    def has_min_time(a: (int, DatagramFrame), b: (int, DatagramFrame)) -> bool:
        return b[1].time_as_timedelta - a[1].time_as_timedelta > min_time

    def get_values(a: (int, DatagramFrame), b: (int, DatagramFrame)) -> tuple[float, float, float]:
        start = a[1].time_as_timedelta.total_seconds()
        end = b[1].time_as_timedelta.total_seconds()
        offset = a[0]
        error = (end - start) / 2
        return start + error, offset, error

    seconds, offsets, errors = unzip3(
        map(lambda t: get_values(*t),
            filter(lambda t: has_min_time(*t),
                   window(
                       add_offset(datagrams)))))

    ax.errorbar(seconds, offsets, xerr=errors, capsize=4, elinewidth=linewidth, color=color, label=label, fmt='none')

    for (x, y, error) in zip(seconds, offsets, errors):
        ms = error * 2 * 1000
        ax.annotate(f'{ms:.2f} ms', xy=(x, y), xytext=(0, -4), textcoords='offset pixels', ha='center', va='top',
                    color=color)

def plot_datagram_ingress_goodput(ax: Axes, client: Connection, chunk_delta: Optional[timedelta] = None):
    """time in s, goodput in Mbps"""
    time, goodputs = unzip(client.ingress_datagram_goodput_chunked(chunk_delta=chunk_delta))
    seconds = [t.total_seconds() for t in time]
    ax.plot(seconds, goodputs)