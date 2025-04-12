#!/usr/bin/env python
# coding: utf-8

import csv
import operator
import re

from array import array
from copy import deepcopy
from pathlib import Path
from typing import Callable, TypeVar, cast
from functools import reduce
from collections.abc import Sequence

type DType = bool | int | float | str
type DFrame = dict[str, Sequence[DType]]

T = TypeVar("T")
V = TypeVar("V")


def _cast_type(x: str) -> DType:
    """Convert a string to a boolean, integer, float, or string."""
    f: Callable[[str], DType] = str

    if re.fullmatch(r"(?i)true|false", x):
        return x.lower() == "true"
    elif re.fullmatch(r"[+-]?\d+", x):
        f = int
    elif re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+)([eE][+-]?\d+)?|[+-]?\d+[eE][+-]?\d+", x):
        f = float

    return f(x)


def _check_column_lengths(df: DFrame) -> None:
    """Check that all columns in the DataFrame have the same length."""
    lengths = {len(col) for col in df.values()}
    if len(lengths) > 1:
        raise ValueError(f"Inconsistent column lengths: {lengths}")


def _array_transformer(d: dict[str, Sequence[DType]]) -> dict[str, Sequence[DType]]:
    """Transform columns of a DataFrame into arrays."""
    d_arrays: dict[str, Sequence[DType]] = {}

    for k, xs in d.items():
        if isinstance(xs, array):
            continue
        if all(isinstance(v, bool) for v in xs):
            d_arrays[k] = array("b", cast(list[int], list(xs)))
        elif all(isinstance(v, int) for v in xs):
            d_arrays[k] = array("i", cast(list[int], list(xs)))
        elif all(isinstance(v, (int, float)) for v in xs):
            d_arrays[k] = array("d", (float(v) for v in list(xs)))
        else:
            d_arrays[k] = xs

    return d_arrays


def _safe_transformer(d: DFrame) -> DFrame:
    """Transform the DataFrame safely, checking column lengths."""
    _check_column_lengths(d)
    return _array_transformer(d)


def from_csv(
    path: Path, has_header: bool = True, encoding: str = "utf8", **kwargs
) -> DFrame:
    """Load a DataFrame from a CSV file."""
    d: dict[str, list[DType]] = {}

    with open(path, mode="r", encoding=encoding) as f:
        for idx, row in enumerate(csv.reader(f, **kwargs)):
            if idx == 0:
                if has_header:
                    for header in row:
                        d[header] = []
                else:
                    for i, v in enumerate(row):
                        d[f"col_{i}"] = [_cast_type(v)]

            else:
                for header, val in zip(d.keys(), row):
                    d[header].append(_cast_type(val))

    return _safe_transformer(cast(DFrame, d))


def columns(df: DFrame) -> set[str]:
    """Return the column names of the DataFrame."""
    return set(df.keys())


def has_column(df: DFrame, col: str) -> bool:
    """Check if a column exists in the DataFrame."""
    return col in columns(df)


def add_column(frame: DFrame, col_name: str, data: Sequence[DType]) -> DFrame:
    """Add a new column to the DataFrame."""
    new_frame = deepcopy(frame)
    new_frame[col_name] = data

    return _safe_transformer(new_frame)


def remove_column(frame: DFrame, col_name: str) -> DFrame:
    """Remove a column from the DataFrame."""
    new_frame = deepcopy(frame)
    if col_name not in frame.keys():
        raise RuntimeError(f"Column {col_name} doesn't exist in DFrame!")

    return {k: v for k, v in new_frame.items() if k != col_name}


def map_column(
    frame: DFrame,
    input_column_name: str,
    output_column_name: str,
    func: Callable[[DType], DType],
) -> DFrame:
    """Apply a function to a column and store the result in a new column."""
    new_frame = deepcopy(frame)

    new_frame[output_column_name] = list(map(func, new_frame[input_column_name]))

    return _safe_transformer(new_frame)


def combine_columns(
    frame: DFrame,
    new_column_name: str,
    left_column_name: str,
    right_column_name: str,
    binary_operator: Callable[[DType, DType], DType],
) -> DFrame:
    """Combine two columns using a binary operator and store the result in a new column."""
    new_frame = deepcopy(frame)
    buffer: list[DType] = []

    for lhs, rhs in zip(new_frame[left_column_name], new_frame[right_column_name]):
        buffer.append(binary_operator(lhs, rhs))

    new_frame[new_column_name] = buffer

    return _safe_transformer(new_frame)


def fold_column(
    frame: DFrame, column_name: str, binary_operator: Callable[[V, DType], V], acc: V
) -> V:
    """Reduce a column to a single value using a binary operator."""
    xs = frame[column_name]
    return reduce(binary_operator, xs, acc)


def sum_column(frame: DFrame, column_name: str) -> float:
    """Calculate the sum of a column."""
    return float(fold_column(
        frame=frame, column_name=column_name, binary_operator=operator.add, acc=0
    ))


def prod_column(frame: DFrame, column_name: str) -> float:
    """Calculate the product of a column."""
    return float(fold_column(
        frame=frame, column_name=column_name, binary_operator=operator.mul, acc=1
    ))


def mean_column(frame: DFrame, column_name: str) -> float:
    """Calculate the mean of a column."""
    return sum_column(frame, column_name) / len(frame[column_name])


def show(frame: DFrame, n_first: int, **kwargs) -> None:
    """Display the first n rows of the DataFrame."""
    cols = columns(frame)

    hs = " | ".join((c for c in cols))
    print(hs, **kwargs)
    print(len(hs) * "-", **kwargs)

    for r in range(n_first):
        print(" | ".join((str(frame[c][r]) for c in cols)), **kwargs)
