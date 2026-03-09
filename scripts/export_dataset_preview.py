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

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, filter_records, record_screen_role, record_workflow, source_index_from_manifest
from roster_taxonomy import source_focus_agent_ids


FIELDS = [
    "record_id",
    "qa_status",
    "source_id",
    "focus_agent_id",
    "workflow",
    "screen_role",
    "head",
    "capture_alias",
    "capture_slot_index",
    "locale",
    "resolution",
    "path",
    "reviewed_uid_full",
    "reviewed_uid_digit",
    "suggested_uid_digit",
    "reviewed_agent_icon_id",
    "suggested_agent_icon_id",
    "reviewed_agent_level",
    "reviewed_agent_level_cap",
    "reviewed_agent_mindscape",
    "reviewed_agent_mindscape_cap",
    "reviewed_agent_stats_json",
    "reviewed_owned_agent_count",
    "reviewed_owned_agent_ids",
    "reviewed_amplifier_id",
    "reviewed_amplifier_name",
    "suggested_amplifier_id",
    "reviewed_amplifier_level",
    "reviewed_amplifier_level_cap",
    "reviewed_amplifier_base_stat_key",
    "reviewed_amplifier_base_stat_value",
    "reviewed_amplifier_advanced_stat_key",
    "reviewed_amplifier_advanced_stat_value",
    "reviewed_disc_set_id",
    "suggested_disc_set_id",
    "reviewed_disc_set_name",
    "reviewed_disc_slot",
    "reviewed_disc_level",
    "suggested_disc_level",
    "reviewed_disc_level_cap",
    "reviewed_disc_main_stat_key",
    "reviewed_disc_main_stat_value",
    "reviewed_disc_substats_json",
    "reviewed_equipment_agent_id",
    "reviewed_equipment_disc_summary_json",
    "suggested_confidence",
]


def _labels(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    labels = record.get(key)
    if not isinstance(labels, dict):
        return {}
    return labels


def _value(labels: Dict[str, Any], key: str) -> str:
    value = labels.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _json_value(labels: Dict[str, Any], key: str) -> str:
    value = labels.get(key)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


def _confidence(labels: Dict[str, Any]) -> str:
    value = labels.get("confidence")
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def _capture_hint(record: Dict[str, Any], key: str) -> str:
    payload = record.get("captureHints")
    if not isinstance(payload, dict):
        return ""
    value = payload.get(key)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return ""


def _joined_list(labels: Dict[str, Any], key: str) -> str:
    values = labels.get(key)
    if not isinstance(values, list):
        return ""
    return "|".join(str(value).strip() for value in values if isinstance(value, str) and str(value).strip())


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
        "workflow": record_workflow(record, source_index),
        "screen_role": record_screen_role(record, source_index),
        "head": head,
        "capture_alias": _capture_hint(record, "screenAlias"),
        "capture_slot_index": _capture_hint(record, "slotIndex"),
        "locale": str(record.get("locale") or ""),
        "resolution": str(record.get("resolution") or ""),
        "path": str(record.get("path") or ""),
        "reviewed_uid_full": _value(reviewed, "uid_full"),
        "reviewed_uid_digit": _value(reviewed, "uid_digit"),
        "suggested_uid_digit": _value(suggested, "uid_digit"),
        "reviewed_agent_icon_id": _value(reviewed, "agent_icon_id"),
        "suggested_agent_icon_id": _value(suggested, "agent_icon_id"),
        "reviewed_agent_level": _value(reviewed, "agent_level"),
        "reviewed_agent_level_cap": _value(reviewed, "agent_level_cap"),
        "reviewed_agent_mindscape": _value(reviewed, "agent_mindscape"),
        "reviewed_agent_mindscape_cap": _value(reviewed, "agent_mindscape_cap"),
        "reviewed_agent_stats_json": _json_value(reviewed, "agent_stats"),
        "reviewed_owned_agent_count": str(len([item for item in reviewed.get("owned_agent_ids", []) if isinstance(item, str) and item.strip()])) if isinstance(reviewed.get("owned_agent_ids"), list) else "",
        "reviewed_owned_agent_ids": _joined_list(reviewed, "owned_agent_ids"),
        "reviewed_amplifier_id": _value(reviewed, "amplifier_id"),
        "reviewed_amplifier_name": _value(reviewed, "amplifier_name"),
        "suggested_amplifier_id": _value(suggested, "amplifier_id"),
        "reviewed_amplifier_level": _value(reviewed, "amplifier_level"),
        "reviewed_amplifier_level_cap": _value(reviewed, "amplifier_level_cap"),
        "reviewed_amplifier_base_stat_key": _value(reviewed, "amplifier_base_stat_key"),
        "reviewed_amplifier_base_stat_value": _value(reviewed, "amplifier_base_stat_value"),
        "reviewed_amplifier_advanced_stat_key": _value(reviewed, "amplifier_advanced_stat_key"),
        "reviewed_amplifier_advanced_stat_value": _value(reviewed, "amplifier_advanced_stat_value"),
        "reviewed_disc_set_id": _value(reviewed, "disc_set_id"),
        "suggested_disc_set_id": _value(suggested, "disc_set_id"),
        "reviewed_disc_set_name": _value(reviewed, "disc_set_name"),
        "reviewed_disc_slot": _value(reviewed, "disc_slot"),
        "reviewed_disc_level": _value(reviewed, "disc_level"),
        "suggested_disc_level": _value(suggested, "disc_level"),
        "reviewed_disc_level_cap": _value(reviewed, "disc_level_cap"),
        "reviewed_disc_main_stat_key": _value(reviewed, "disc_main_stat_key"),
        "reviewed_disc_main_stat_value": _value(reviewed, "disc_main_stat_value"),
        "reviewed_disc_substats_json": _json_value(reviewed, "disc_substats"),
        "reviewed_equipment_agent_id": _value(reviewed, "equipment_agent_id"),
        "reviewed_equipment_disc_summary_json": _json_value(reviewed, "equipment_disc_summary"),
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
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true", default=True)
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    sources = manifest.get("sources", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = source_index_from_manifest(sources)
    scoped_records = filter_records(
        records,
        source_index=source_index,
        workflow=args.workflow,
        import_eligible_only=bool(args.import_eligible_only),
    )
    selected = _select_records(scoped_records, status_filter=args.status, max_records=max(0, int(args.max_records)))
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
