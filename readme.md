# funFrame – A Minimal DataFrame Library in Pure Python 🐍

**funFrame** is a small educational project that implements a DataFrame-style API using only Python's standard library.

This project was built for learning and experimentation. It is **not optimized for performance** and should not be used in production scenarios. The goal was to explore functional programming concepts and build a flexible, testable data manipulation library from scratch — without relying on external libraries like `pandas` or `numpy`.

## ✨ Features

- Pure Python: built using only the standard library (e.g., `csv`, `array`, `copy`, `re`)
- A functional core API (pure functions for transformations)
- A convenient object-oriented wrapper: `DataFrame` class
- Type-safe with support for `int`, `float`, `bool`, and `str` data types
- Read and parse CSV files
- Transform, filter, and combine columns
- Perform simple aggregations like `sum`, `product`, and `mean`

## 📦 Installation

```bash
git clone https://github.com/roman91DE/funFrame.git
```

There are no external dependencies!

## 🚀 Example Usage

```python
from funframe import DataFrame

# Create a DataFrame from a Python dict
df = DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": ["x", "y", "z"]
})

# Add a new column
df2 = df.add_column("d", [7, 8, 9])

# Combine columns
df3 = df2.combine_columns("sum_ab", "a", "b", lambda x, y: x + y)

# Filter rows
filtered = df3.filter_on_column("a", lambda x: x > 1)

# Show data
filtered.show_all()
```

## 📁 CSV Support

```python
from pathlib import Path

df = DataFrame.from_csv(Path("data.csv"))
df.show(5)
```

Supports:
- Optional headers (`has_header=False`)
- Custom delimiters (`delimiter=";"`, `delimiter="?"`, etc.)
- Automatic type inference

## 🧪 Tests

The project comes with a suite of unit tests covering both functional and object-oriented APIs:

```bash
python test_funframe.py
```

## ⚠️ Disclaimer

This project is intentionally **simple and minimal**:
- It's not fast (arrays are used, but no vectorization)
- It’s not comprehensive (no missing value support, no grouping, no joins)
- It's for learning and prototyping only

## 📚 Philosophy

The implementation focuses on:
- Pure functions (where possible)
- Copy-on-write semantics (`deepcopy`)
- Transparent data structures (`dict[str, list|array]`)
- Explicit error handling and validation

## 🧠 You Might Like This If You...

- Want to understand how libraries like `pandas` work under the hood
- Are learning Python and want a hands-on project
- Like building things from scratch instead of relying on magic ✨

