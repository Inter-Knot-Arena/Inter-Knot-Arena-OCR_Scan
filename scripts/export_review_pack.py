from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest


def _get_label(labels: Dict[str, Any], key: str) -> str:
    value = labels.get(key)
    return str(value) if isinstance(value, str) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OCR review queue CSV from dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-csv", default="docs/review_queue.csv")
    parser.add_argument("--status", default="needs_review", help="qaStatus filter, e.g. needs_review|unlabeled|any")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    status_filter = str(args.status).strip().lower()
    selected: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        status = str(record.get("qaStatus") or "").strip().lower()
        if status_filter != "any" and status != status_filter:
            continue
        selected.append(record)
        if args.max_records > 0 and len(selected) >= args.max_records:
            break

    output_path = Path(args.output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "record_id",
                "path",
                "head",
                "locale",
                "resolution",
                "uid_digit",
                "agent_icon_id",
                "amplifier_id",
                "disc_set_id",
                "disc_level",
                "reviewer",
                "notes",
            ],
        )
        writer.writeheader()
        for record in selected:
            labels = record.get("labels")
            if not isinstance(labels, dict):
                labels = {}
            writer.writerow(
                {
                    "record_id": str(record.get("id") or ""),
                    "path": str(record.get("path") or ""),
                    "head": str(record.get("head") or labels.get("head") or ""),
                    "locale": str(record.get("locale") or ""),
                    "resolution": str(record.get("resolution") or ""),
                    "uid_digit": _get_label(labels, "uid_digit"),
                    "agent_icon_id": _get_label(labels, "agent_icon_id"),
                    "amplifier_id": _get_label(labels, "amplifier_id"),
                    "disc_set_id": _get_label(labels, "disc_set_id"),
                    "disc_level": _get_label(labels, "disc_level"),
                    "reviewer": "",
                    "notes": "",
                }
            )

    print(f"Exported {len(selected)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

