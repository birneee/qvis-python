import sys
import os

sys.path.append(os.getcwd())

import asyncio
import ujson

import cysimdjson

from qvis.utils.time_utils import print_func_time
import qvis.event
from qvis.connection import parse_qlog


class Test:

    def __init__(self):
        pass

    def parse(self, line: str):
        pass

    def __call__(self, line):
        return self.parse(line)


async def process(parser: cysimdjson.JSONParser, i: asyncio.Queue, o: asyncio.Queue):
    line = await i.get()
    json = parser.parse(line)
    await o.put(json)


@print_func_time
def readLines():
    r = open('testdata/client.qlog', "rb")
    line = r.readline()
    while line:
        line = r.readline()


@print_func_time
def parseWithSimdJson():
    parser = cysimdjson.JSONParser()
    r = open('testdata/client.qlog', "rb")
    line = r.readline()
    while line:
        if line.startswith(b'{'):  # if json
            yield parser.parse(line)
        line = r.readline()


@print_func_time
def parseWithUJson():
    r = open('testdata/client.qlog', "rb")
    line = r.readline()
    while line:
        if line.startswith(b'{'):  # if json
            ujson.loads(line)
        line = r.readline()


@print_func_time
def iterateEventsWithUJson():
    r = open('testdata/client.qlog', "rb")
    r.readline()  # skip header
    line = r.readline()
    while line:
        if line.startswith(b'{'):  # if json
            json = ujson.loads(line)
            qvis.event.Event(json, None, r.tell())
        line = r.readline()


@print_func_time
def iterateEventsWithSimdJson():
    parser = cysimdjson.JSONParser()
    r = open('testdata/client.qlog', "rb")
    r.readline()  # skip header
    line = r.readline()
    while line:
        if line.startswith(b'{'):  # if json
            json = parser.parse(line)
            qvis.event.Event(json, None, r.tell())
        line = r.readline()


@print_func_time
def iterateReceivedFrames():
    r = open('testdata/client.qlog', "rb")
    qlog = parse_qlog(r)
    for stream_frame in qlog.received_stream_frames_of_stream(3):
        pass


def main():
    readLines()
    parseWithSimdJson()
    parseWithUJson()
    iterateEventsWithUJson()
    iterateEventsWithSimdJson()
    iterateReceivedFrames()


if __name__ == "__main__":
    main()
