"""A tiny calculator with one deliberate bug for the mca demo.

The bug is in ``subtract``: it adds instead of subtracts. ``tests/test_calculator.py``
fails until that one line is corrected.
"""

from __future__ import annotations


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a + b  # BUG: should be a - b


def multiply(a: float, b: float) -> float:
    return a * b
