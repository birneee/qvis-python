#!/usr/bin/env python

import json
import pandas as pd


def pandas_read_qlog(filepath: str) -> pd.DataFrame:
    df = pd.DataFrame()
    with open(filepath) as file:
        for line in file:
            line
            #data = json.load(line)
    return df