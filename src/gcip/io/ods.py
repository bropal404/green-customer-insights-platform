"""Streaming reader for OpenDocument spreadsheets (.ods).

DfT and DESNZ publish multi-megabyte .ods files; parsing them with odfpy needs
minutes and gigabytes. An .ods file is a zip holding ``content.xml``, so we
stream that XML with ``lxml.etree.iterparse`` and keep only the table we need.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
from lxml import etree

_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_T = "{%s}" % _TABLE_NS
_O = "{%s}" % _OFFICE_NS

# Repeated empty cells at the end of a row can be declared as e.g. 16,000 copies;
# never expand more than this many columns.
_MAX_COLUMNS = 512


def _cell_value(cell: etree._Element) -> str | float | None:
    value_type = cell.get(_O + "value-type")
    if value_type in ("float", "percentage", "currency"):
        raw = cell.get(_O + "value")
        return float(raw) if raw is not None else None
    if value_type == "date":
        return cell.get(_O + "date-value")
    text = "".join(cell.itertext()).strip()
    return text if text else None


def read_ods_table(path: str | Path, table_name: str, header_row_contains: str | None = None) -> pd.DataFrame:
    """Return one sheet of an .ods file as a DataFrame of raw cell values.

    If ``header_row_contains`` is given, the first row whose first cell equals that
    text becomes the header and earlier rows (titles, notes) are dropped.
    """
    rows: list[list] = []
    in_table = False
    with zipfile.ZipFile(path) as archive, archive.open("content.xml") as handle:
        for event, element in etree.iterparse(handle, events=("start", "end")):
            tag = element.tag
            if event == "start" and tag == _T + "table":
                in_table = element.get(_T + "name") == table_name
                continue
            if event != "end":
                continue
            if tag == _T + "table-row":
                if in_table:
                    row: list = []
                    for cell in element.iterchildren(_T + "table-cell", _T + "covered-table-cell"):
                        repeat = int(cell.get(_T + "number-columns-repeated", "1"))
                        value = _cell_value(cell)
                        row.extend([value] * min(repeat, _MAX_COLUMNS - len(row)))
                        if len(row) >= _MAX_COLUMNS:
                            break
                    repeat_rows = int(element.get(_T + "number-rows-repeated", "1"))
                    if any(v is not None for v in row):
                        rows.extend([row] * min(repeat_rows, 1))
                element.clear()
            elif tag == _T + "table":
                if in_table:
                    break
                element.clear()
    if not rows:
        raise ValueError(f"table {table_name!r} not found or empty in {path}")
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    frame = pd.DataFrame(rows)
    if header_row_contains is not None:
        matches = frame.index[frame.iloc[:, 0].astype(str).str.strip() == header_row_contains]
        if len(matches) == 0:
            raise ValueError(f"header row starting with {header_row_contains!r} not found in {table_name}")
        start = matches[0]
        header = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(frame.iloc[start])]
        frame = frame.iloc[start + 1 :].reset_index(drop=True)
        frame.columns = header
        frame = frame.loc[:, [c for c in frame.columns if not c.startswith("col_")]]
    return frame
