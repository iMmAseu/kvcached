#!/usr/bin/env python3
"""
Extract the first worksheet of an .xlsx file into CSV without third-party deps.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx


def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    ns = {"a": NS_MAIN}
    values: list[str] = []
    for si in root.findall("a:si", ns):
        text = "".join(t.text or "" for t in si.findall(".//a:t", ns))
        values.append(text)
    return values


def _first_sheet_target(zf: zipfile.ZipFile) -> str:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    ns = {"a": NS_MAIN}

    rid_to_target = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    sheets = wb.find("a:sheets", ns)
    if sheets is None or len(sheets) == 0:
        raise RuntimeError("No worksheet found in workbook.")

    first = sheets[0]
    rid = first.attrib.get(f"{{{NS_REL}}}id")
    if not rid or rid not in rid_to_target:
        raise RuntimeError("Cannot resolve first worksheet relationship.")

    target = rid_to_target[rid].lstrip("/")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target


def extract_to_rows(xlsx_path: Path) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared = _load_shared_strings(zf)
        sheet_xml = _first_sheet_target(zf)
        root = ET.fromstring(zf.read(sheet_xml))
        ns = {"a": NS_MAIN}

        data = root.find("a:sheetData", ns)
        if data is None:
            return []

        row_maps: list[dict[int, str]] = []
        max_col = 0

        for row in data.findall("a:row", ns):
            row_map: dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                col = _col_index(ref) if ref else 0
                if col <= 0:
                    continue

                cell_type = cell.attrib.get("t")
                v = cell.find("a:v", ns)
                if v is None:
                    text = ""
                else:
                    raw = v.text or ""
                    if cell_type == "s":
                        if raw.isdigit():
                            si = int(raw)
                            text = shared[si] if 0 <= si < len(shared) else raw
                        else:
                            text = raw
                    else:
                        text = raw

                row_map[col] = text
                if col > max_col:
                    max_col = col
            row_maps.append(row_map)

        rows: list[list[str]] = []
        for rm in row_maps:
            rows.append([rm.get(i, "") for i in range(1, max_col + 1)])
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract first .xlsx sheet to CSV using stdlib only.")
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("csv_out", type=Path)
    args = parser.parse_args()

    if not args.xlsx.exists():
        print(f"Input not found: {args.xlsx}", file=sys.stderr)
        return 2

    rows = extract_to_rows(args.xlsx)
    if not rows:
        print("No data extracted from workbook.", file=sys.stderr)
        return 3

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
