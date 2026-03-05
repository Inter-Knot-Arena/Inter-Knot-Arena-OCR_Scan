from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import source_focus_agent_ids


FIELDS = [
    "record_id",
    "qa_status",
    "source_id",
    "focus_agent_id",
    "head",
    "locale",
    "resolution",
    "path",
    "reviewed_uid_digit",
    "suggested_uid_digit",
    "reviewed_agent_icon_id",
    "suggested_agent_icon_id",
    "reviewed_amplifier_id",
    "suggested_amplifier_id",
    "reviewed_disc_set_id",
    "suggested_disc_set_id",
    "reviewed_disc_level",
    "suggested_disc_level",
    "suggested_confidence",
]


def _labels(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    labels = record.get(key)
    if not isinstance(labels, dict):
        return {}
    return labels


def _value(labels: Dict[str, Any], key: str) -> str:
    value = labels.get(key)
    return str(value) if isinstance(value, str) else ""


def _confidence(labels: Dict[str, Any]) -> str:
    value = labels.get("confidence")
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def _source_focus(source_index: Dict[str, Dict[str, Any]], source_id: str) -> str:
    source = source_index.get(source_id, {})
    focus = source_focus_agent_ids(source)
    return focus[0] if focus else ""


def _select_records(records: List[Dict[str, Any]], status_filter: str, max_records: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    normalized_status = status_filter.strip().lower()
    ordered = sorted(records, key=lambda item: (str(item.get("qaStatus") or ""), str(item.get("id") or "")))
    for record in ordered:
        if not isinstance(record, dict):
            continue
        status = str(record.get("qaStatus") or "").strip().lower()
        if normalized_status != "any" and status != normalized_status:
            continue
        selected.append(record)
        if max_records > 0 and len(selected) >= max_records:
            break
    return selected


def _row(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    reviewed = _labels(record, "labels")
    suggested = _labels(record, "suggestedLabels")
    source_id = str(record.get("sourceId") or "src_unknown")
    head = str(record.get("head") or reviewed.get("head") or suggested.get("head") or "")
    return {
        "record_id": str(record.get("id") or ""),
        "qa_status": str(record.get("qaStatus") or ""),
        "source_id": source_id,
        "focus_agent_id": _source_focus(source_index, source_id),
        "head": head,
        "locale": str(record.get("locale") or ""),
        "resolution": str(record.get("resolution") or ""),
        "path": str(record.get("path") or ""),
        "reviewed_uid_digit": _value(reviewed, "uid_digit"),
        "suggested_uid_digit": _value(suggested, "uid_digit"),
        "reviewed_agent_icon_id": _value(reviewed, "agent_icon_id"),
        "suggested_agent_icon_id": _value(suggested, "agent_icon_id"),
        "reviewed_amplifier_id": _value(reviewed, "amplifier_id"),
        "suggested_amplifier_id": _value(suggested, "amplifier_id"),
        "reviewed_disc_set_id": _value(reviewed, "disc_set_id"),
        "suggested_disc_set_id": _value(suggested, "disc_set_id"),
        "reviewed_disc_level": _value(reviewed, "disc_level"),
        "suggested_disc_level": _value(suggested, "disc_level"),
        "suggested_confidence": _confidence(suggested),
    }


def _write_html(rows: List[Dict[str, str]], output_path: Path) -> None:
    headers = "".join(f"<th>{html.escape(field)}</th>" for field in FIELDS)
    body_rows = []
    for row in rows:
        cells: List[str] = []
        for field in FIELDS:
            value = row.get(field, "")
            if field == "path" and value:
                uri = Path(value).resolve().as_uri()
                display = html.escape(value)
                cell = f'<a href="{uri}">{display}</a><br><img src="{uri}" style="max-width:240px;max-height:140px;object-fit:contain;background:#111">'
            else:
                cell = html.escape(value)
            cells.append(f"<td>{cell}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    document = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>OCR Dataset Preview</title>
<style>
body {{ font-family: Segoe UI, sans-serif; background: #111; color: #eee; margin: 20px; }}
a {{ color: #9bd; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #333; padding: 8px; vertical-align: top; font-size: 12px; }}
th {{ position: sticky; top: 0; background: #1a1a1a; }}
tr:nth-child(even) {{ background: #171717; }}
</style>
</head>
<body>
<h1>OCR Dataset Preview</h1>
<p>Rows: {len(rows)}</p>
<table>
<thead><tr>{headers}</tr></thead>
<tbody>
{''.join(body_rows)}
</tbody>
</table>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OCR dataset preview with reviewed and suggested labels.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-csv", default="docs/dataset_preview.csv")
    parser.add_argument("--output-html", default="docs/dataset_preview.html")
    parser.add_argument("--status", default="any")
    parser.add_argument("--max-records", type=int, default=450)
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    sources = manifest.get("sources", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = {
        str(source.get("sourceId") or ""): source
        for source in sources
        if isinstance(source, dict) and str(source.get("sourceId") or "")
    }
    selected = _select_records(records, status_filter=args.status, max_records=max(0, int(args.max_records)))
    rows = [_row(record, source_index) for record in selected]

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    output_html = Path(args.output_html).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    _write_html(rows, output_html)

    print(json.dumps({"rows": len(rows), "csv": str(output_csv), "html": str(output_html)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
