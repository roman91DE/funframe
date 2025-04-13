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
type _DFrame = dict[str, Sequence[DType]]

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


def _check_column_lengths(df: _DFrame) -> None:
    """Check that all columns in the DataFrame have the same length."""
    lengths = {len(col) for col in df.values()}
    if len(lengths) > 1:
        raise ValueError(f"Inconsistent column lengths: {lengths}")


def _array_transformer(d: dict[str, Sequence[DType]]) -> dict[str, Sequence[DType]]:
    """Transform columns of a DataFrame into arrays."""
    d_arrays: dict[str, Sequence[DType]] = {}

    for k, xs in d.items():
        if isinstance(xs, array):
            d_arrays[k] = xs
            continue
        elif all(isinstance(v, bool) for v in xs):
            d_arrays[k] = array("b", cast(list[int], list(xs)))
        elif all(isinstance(v, int) for v in xs):
            d_arrays[k] = array("i", cast(list[int], list(xs)))
        elif all(isinstance(v, (int, float)) for v in xs):
            d_arrays[k] = array("d", (float(v) for v in list(xs)))
        else:
            d_arrays[k] = xs

    return d_arrays


def _safe_transformer(d: _DFrame) -> _DFrame:
    """Transform the DataFrame safely, checking column lengths."""
    _check_column_lengths(d)
    return _array_transformer(d)


def _from_csv(
    path: Path, has_header: bool = True, encoding: str = "utf8", **kwargs
) -> _DFrame:
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

    return _safe_transformer(cast(_DFrame, d))


def _columns(df: _DFrame) -> set[str]:
    """Return the column names of the DataFrame."""
    return set(df.keys())


def _has_column(df: _DFrame, col: str) -> bool:
    """Check if a column exists in the DataFrame."""
    return col in _columns(df)


def _add_column(frame: _DFrame, col_name: str, data: Sequence[DType]) -> _DFrame:
    """Add a new column to the DataFrame."""
    new_frame = deepcopy(frame)
    new_frame[col_name] = data

    return _safe_transformer(new_frame)


def _remove_column(frame: _DFrame, col_name: str) -> _DFrame:
    """Remove a column from the DataFrame."""
    new_frame = deepcopy(frame)
    if col_name not in frame.keys():
        raise RuntimeError(f"Column {col_name} doesn't exist in DFrame!")

    return {k: v for k, v in new_frame.items() if k != col_name}


def _map_column(
    frame: _DFrame,
    input_column_name: str,
    output_column_name: str,
    func: Callable[[DType], DType],
) -> _DFrame:
    """Apply a function to a column and store the result in a new column."""
    new_frame = deepcopy(frame)

    new_frame[output_column_name] = list(map(func, new_frame[input_column_name]))

    return _safe_transformer(new_frame)


def _select_rows(frame: _DFrame, indices: Sequence[int]) -> _DFrame:
    """Select specific rows from the DataFrame."""
    new_frame: _DFrame = {str(k): [] for k in frame.keys()}

    for col in frame.keys():
        buf = []
        for idx, val in enumerate(frame[col]):
            if idx in indices:
                buf.append(val)
        new_frame[col] = buf

    return _safe_transformer(new_frame)


def _filter_on_column(
    frame: _DFrame,
    column_name: str,
    predicate_func: Callable[[DType], bool],
) -> _DFrame:
    """Filter rows based on a predicate function applied to a column."""

    indices: list[int] = []
    for i, v in enumerate(frame[column_name]):
        if predicate_func(v):
            indices.append(i)

    return _select_rows(frame, indices)


def _combine_columns(
    frame: _DFrame,
    new_column_name: str,
    left_column_name: str,
    right_column_name: str,
    binary_operator: Callable[[DType, DType], DType],
) -> _DFrame:
    """Combine two columns using a binary operator and store the result in a new column."""
    new_frame = deepcopy(frame)
    buffer: list[DType] = []

    for lhs, rhs in zip(new_frame[left_column_name], new_frame[right_column_name]):
        buffer.append(binary_operator(lhs, rhs))

    new_frame[new_column_name] = buffer

    return _safe_transformer(new_frame)


def _fold_column(
    frame: _DFrame, column_name: str, binary_operator: Callable[[V, DType], V], acc: V
) -> V:
    """Reduce a column to a single value using a binary operator."""
    xs = frame[column_name]
    return reduce(binary_operator, xs, acc)


def _sum_column(frame: _DFrame, column_name: str) -> float:
    """Calculate the sum of a column."""
    return float(
        _fold_column(
            frame=frame, column_name=column_name, binary_operator=operator.add, acc=0
        )
    )


def _prod_column(frame: _DFrame, column_name: str) -> float:
    """Calculate the product of a column."""
    return float(
        _fold_column(
            frame=frame, column_name=column_name, binary_operator=operator.mul, acc=1
        )
    )


def _mean_column(frame: _DFrame, column_name: str) -> float:
    """Calculate the mean of a column."""
    return _sum_column(frame, column_name) / len(frame[column_name])


def _show(frame: _DFrame, n_first: int, **kwargs) -> None:
    """Display the first n rows of the DataFrame with aligned columns."""
    cols = list(_columns(frame))

    col_widths = {
        col: max(
            len(col),
            *(len(str(frame[col][i])) for i in range(min(n_first, len(frame[col])))),
        )
        for col in cols
    }

    row_fmt = " | ".join(f"{{:{col_widths[col]}}}" for col in cols)

    print(row_fmt.format(*cols), **kwargs)
    print("-+-".join("-" * col_widths[col] for col in cols), **kwargs)

    for r in range(n_first):
        print(row_fmt.format(*(str(frame[c][r]) for c in cols)), **kwargs)


def _show_all(frame: _DFrame, **kwargs) -> None:
    """Display all rows of the DataFrame with aligned columns."""
    _, n = _shape(frame)
    _show(frame=frame, n_first=n, **kwargs)


def _shape(frame: _DFrame) -> tuple[int, int]:
    """Return the shape of the DataFrame."""
    ncols = len(frame)
    for _, v in frame.items():
        nrows = len(v)
        break
    return (ncols, nrows)


class DataFrame:
    def __init__(self, data: _DFrame) -> None:
        """Initialize the DataFrame with validated data."""
        _check_column_lengths(data)
        self._data = _safe_transformer(data)

    @classmethod
    def from_csv(
        cls, path: Path, has_header: bool = True, encoding: str = "utf8", **kwargs
    ) -> "DataFrame":
        """
        Create a DataFrame from a CSV file.

        Args:
            path: Path to the CSV file.
            has_header: Whether the CSV file has a header row.
            encoding: Encoding of the CSV file.
            **kwargs: Additional arguments to pass to the CSV reader.

        Returns:
            A new DataFrame instance.
        """
        return cls(_from_csv(path, has_header, encoding, **kwargs))

    def columns(self) -> set[str]:
        """
        Return the column names of the DataFrame.

        Returns:
            A set of column names.
        """
        return _columns(self._data)

    def has_column(self, col: str) -> bool:
        """
        Check if a column exists in the DataFrame.

        Args:
            col: Column name to check.

        Returns:
            True if the column exists, False otherwise.
        """
        return _has_column(self._data, col)

    def add_column(self, col_name: str, data: Sequence[DType]) -> "DataFrame":
        """
        Add a new column to the DataFrame.

        Args:
            col_name: Name of the new column.
            data: Data for the new column.

        Returns:
            A new DataFrame with the added column.
        """
        return DataFrame(_add_column(self._data, col_name, data))

    def remove_column(self, col_name: str) -> "DataFrame":
        """
        Remove a column from the DataFrame.

        Args:
            col_name: Name of the column to remove.

        Returns:
            A new DataFrame without the specified column.
        """
        return DataFrame(_remove_column(self._data, col_name))

    def map_column(
        self,
        input_column_name: str,
        output_column_name: str,
        func: Callable[[DType], DType],
    ) -> "DataFrame":
        """
        Apply a function to a column and store the result in a new column.

        Args:
            input_column_name: Name of the input column.
            output_column_name: Name of the new output column.
            func: Function to apply to each element of the input column.

        Returns:
            A new DataFrame with the added output column.
        """
        return DataFrame(
            _map_column(self._data, input_column_name, output_column_name, func)
        )

    def select_rows(self, indices: Sequence[int]) -> "DataFrame":
        """
        Select specific rows from the DataFrame.

        Args:
            indices: Indices of rows to select.

        Returns:
            A new DataFrame containing only the selected rows.
        """
        return DataFrame(_select_rows(self._data, indices))

    def filter_on_column(
        self, column_name: str, predicate_func: Callable[[DType], bool]
    ) -> "DataFrame":
        """
        Filter rows based on a predicate function applied to a column.

        Args:
            column_name: Column to apply the predicate on.
            predicate_func: Function that returns True for rows to keep.

        Returns:
            A new DataFrame with filtered rows.
        """
        return DataFrame(_filter_on_column(self._data, column_name, predicate_func))

    def combine_columns(
        self,
        new_column_name: str,
        left_column_name: str,
        right_column_name: str,
        binary_operator: Callable[[DType, DType], DType],
    ) -> "DataFrame":
        """
        Combine two columns using a binary operator and store the result in a new column.

        Args:
            new_column_name: Name of the new column.
            left_column_name: Name of the left input column.
            right_column_name: Name of the right input column.
            binary_operator: Function to combine two elements.

        Returns:
            A new DataFrame with the combined column.
        """
        return DataFrame(
            _combine_columns(
                self._data,
                new_column_name,
                left_column_name,
                right_column_name,
                binary_operator,
            )
        )

    def fold_column(
        self, column_name: str, binary_operator: Callable[[V, DType], V], acc: V
    ) -> V:
        """
        Reduce a column to a single value using a binary operator.

        Args:
            column_name: Name of the column to reduce.
            binary_operator: Function to combine the accumulated result with a value.
            acc: Initial accumulator value.

        Returns:
            The final reduced value.
        """
        return _fold_column(self._data, column_name, binary_operator, acc)

    def sum_column(self, column_name: str) -> float:
        """
        Calculate the sum of a column.

        Args:
            column_name: Name of the column.

        Returns:
            Sum of the column as a float.
        """
        return _sum_column(self._data, column_name)

    def prod_column(self, column_name: str) -> float:
        """
        Calculate the product of a column.

        Args:
            column_name: Name of the column.

        Returns:
            Product of the column as a float.
        """
        return _prod_column(self._data, column_name)

    def mean_column(self, column_name: str) -> float:
        """
        Calculate the mean of a column.

        Args:
            column_name: Name of the column.

        Returns:
            Mean of the column as a float.
        """
        return _mean_column(self._data, column_name)

    def show(self, n_first: int, **kwargs) -> None:
        """
        Display the first n rows of the DataFrame.

        Args:
            n_first: Number of rows to display.
            **kwargs: Additional arguments to pass to the print function.
        """
        _show(self._data, n_first, **kwargs)

    def show_all(self, **kwargs) -> None:
        """
        Display all rows of the DataFrame.

        Args:
            **kwargs: Additional arguments to pass to the print function.
        """
        _show_all(self._data, **kwargs)

    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the DataFrame.

        Returns:
            A tuple (number of columns, number of rows).
        """
        return _shape(self._data)

    def to_dict(self) -> _DFrame:
        """
        Convert the DataFrame to a dictionary.

        Returns:
            A deep copy of the underlying dictionary.
        """
        return deepcopy(self._data)
