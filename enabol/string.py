from __future__ import annotations

import re
from typing import Any, SupportsIndex

import numpy as np


_ANSI_RESET = "\x1B[0m"
_ANSI_TRUE = "\x1B[38;2;0;204;0m"
_ANSI_FALSE = "\x1B[38;2;204;0;0m"
_ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def format_value(value: Any, *, sig: int = 5) -> str:
    """Compact human-readable formatter for repr tables."""

    if isinstance(value, bool):
        return f"{_ANSI_TRUE}● True{_ANSI_RESET}" if value else f"{_ANSI_FALSE}● False{_ANSI_RESET}"
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(item, sig=sig) for item in value) + "]"
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        if value.size > 6:
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        return "[" + ", ".join(format_value(item, sig=sig) for item in value.flatten()) + "]"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if x == 0:
            return "0"
        return f"{x:.{sig}g}"
    return str(value)


def ansi_length(text: str) -> int:
    """Length after stripping ANSI escape sequences."""

    return len(_ANSI_ESCAPE.sub("", text))


class AnsiString(str):
    """String with ANSI-aware justification helpers."""

    def __new__(cls, content: Any):
        text = str(content)
        obj = super().__new__(cls, text)
        obj.len = ansi_length(text)
        return obj

    def ljust(self, width: SupportsIndex, fillchar: str = " ") -> str:
        pad = int(width) - self.len
        if pad > 0:
            return str(self) + fillchar * pad
        return str(self)

    def rjust(self, width: SupportsIndex, fillchar: str = " ") -> str:
        pad = int(width) - self.len
        if pad > 0:
            return fillchar * pad + str(self)
        return str(self)


def _cell_lines(value: Any) -> list[AnsiString]:
    text = "" if value is None else format_value(value)
    return [AnsiString(line) for line in (text.splitlines() or [""])]


def print_table(data: dict[str, Any], header: str, *, key_header: str = "Field") -> str:
    """Render a two-column or multi-column Unicode table.

    Values can be scalars, multi-line strings, or dictionaries. A ``None``
    value marks a section header row.
    """

    any_dict = any(isinstance(value, dict) for value in data.values() if value is not None)
    if not any_dict:
        rows = [
            (AnsiString(format_value(key)), None if value is None else {"value": value})
            for key, value in data.items()
        ]
        key_w = max((key.len for key, _ in rows), default=0)
        value_w = max(
            (
                line.len
                for _, row in rows
                if row is not None
                for line in _cell_lines(row["value"])
            ),
            default=0,
        )
        segments = [key_w + 2, value_w + 2]
        return _render_table(header, key_header, rows, ("value",), {"value": value_w}, segments)

    normalized: list[tuple[AnsiString, dict[str, Any] | None]] = []
    columns: list[str] = []
    column_seen: set[str] = set()
    for key, value in data.items():
        row_key = AnsiString(format_value(key))
        if value is None:
            normalized.append((row_key, None))
            continue
        row = value if isinstance(value, dict) else {"value": value}
        normalized.append((row_key, row))
        for column in row:
            if column not in column_seen:
                column_seen.add(column)
                columns.append(column)

    key_w = max((key.len for key, _ in normalized), default=0)
    column_widths: dict[str, int] = {}
    for column in columns:
        width = AnsiString(format_value(column)).len
        for _, row in normalized:
            if row is None:
                continue
            for line in _cell_lines(row.get(column, "")):
                width = max(width, line.len)
        column_widths[column] = width

    segments = [key_w + 2] + [column_widths[column] + 2 for column in columns]
    return _render_table(header, key_header, normalized, tuple(columns), column_widths, segments)


def _rule(left: str, mid: str, right: str, segments: list[int], fill: str) -> str:
    return left + mid.join(fill * segment for segment in segments) + right + "\n"


def _render_table(
    header: str,
    key_header: str,
    rows: list[tuple[AnsiString, Any]],
    columns: tuple[str, ...],
    column_widths: dict[str, int],
    segments: list[int],
) -> str:
    header_a = AnsiString(header)
    key_w = segments[0] - 2
    inner_w = sum(segments) + (len(segments) - 1)

    text = "╔" + "═" * inner_w + "╗\n"
    text += "║" + f" {header_a.ljust(inner_w - 2)} " + "║\n"
    text += _rule("╟", "┬", "╢", segments, "─")

    text += "║"
    text += f" {AnsiString(key_header).ljust(key_w)} "
    for column in columns:
        text += "│"
        text += f" {AnsiString(format_value(column)).ljust(column_widths[column])} "
    text += "║\n"
    text += _rule("╟", "┼", "╢", segments, "─")

    at_boundary = True
    for key, row in rows:
        if row is None:
            if not at_boundary:
                text += _rule("╟", "┴", "╢", segments, "─")
            text += "║" + f" {key.ljust(inner_w - 2)} " + "║\n"
            text += _rule("╟", "┬", "╢", segments, "─")
            at_boundary = True
            continue

        row_dict = row if isinstance(row, dict) else {"value": row}
        column_lines = {column: _cell_lines(row_dict.get(column, "")) for column in columns}
        row_height = max((len(lines) for lines in column_lines.values()), default=1)
        for line_idx in range(row_height):
            text += "║"
            text += f" {(key if line_idx == 0 else AnsiString('')).ljust(key_w)} "
            for column in columns:
                lines = column_lines[column]
                cell = lines[line_idx] if line_idx < len(lines) else AnsiString("")
                text += "│"
                text += f" {cell.ljust(column_widths[column])} "
            text += "║\n"
        at_boundary = False

    text += _rule("╚", "╧", "╝", segments, "═")
    return text
