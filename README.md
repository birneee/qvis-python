# qvis-python

A set of QUIC visualization tools inspired by [qvis](https://github.com/quiclog/qvis).

TODO

## Examples

### Get received bytes on stream 3

```python
from qvis.connection import read_qlog
conn = read_qlog('testdata/client.qlog')
conn.total_received_stream_payload(3)
```

### Plot received bytes on stream 3

```python
from qvis.connection import read_qlog
from qvis.plot import plot_stream_data_received
from matplotlib import pyplot as plt
conn = read_qlog('testdata/client.qlog')
fig, ax = plt.subplots()
plot_stream_data_received(ax, conn, 3)
fig.show()
```