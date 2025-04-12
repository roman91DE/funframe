#!/usr/bin/env python
# coding: utf-8

# In[132]:


import csv
import re

from array import array
from copy import deepcopy
from pathlib import Path
from typing import Iterable


# In[135]:


type DType = bool | int | float | str
type DFrame = dict[str, Iterable]


# In[ ]:


def cast_type(x: str) -> DType:
    f = str

    if re.fullmatch(r"(?i)true|false", x):
        return x.lower() == "true"
    elif re.fullmatch(r"[+-]?\d+", x):
        f = int
    elif re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+", x):
        f = float

    return f(x)


def parse_csv(
    path: Path, has_header: bool = True, encoding: str = "utf8", **kwargs
) -> DFrame:
    d = {}

    with open(path, mode="r", encoding=encoding) as f:
        for idx, row in enumerate(csv.reader(f, **kwargs)):
            if idx == 0:
                if has_header:
                    for header in row:
                        d[header] = []
                else:
                    for i, v in enumerate(row):
                        d[f"col_{i}"] = [cast_type(v)]

            else:
                for header, val in zip(d.keys(), row):
                    d[header].append(cast_type(val))
    d_arrays = {}

    for k, xs in d.items():
        if all(isinstance(v, bool) for v in xs):
            d_arrays[k] = array("b", xs)
        elif all(isinstance(v, int) for v in xs):
            d_arrays[k] = array("i", xs)
        elif all(isinstance(v, (int, float)) for v in xs):
            d_arrays[k] = array("d", (float(v) for v in xs))
        else:
            d_arrays[k] = xs

    return d_arrays



def columns(df: DFrame) -> set[str]:
    return set(df.keys())


def has_column(df: DFrame, col: str) -> bool:
    return col in columns(df)



def add_column(frame: DFrame, col_name: str, data: Iterable) -> DFrame:
    new_frame = deepcopy(frame)
    new_frame[col_name] = data
    return new_frame

