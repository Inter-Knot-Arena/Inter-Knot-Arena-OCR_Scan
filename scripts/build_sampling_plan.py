from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from manifest_lib import ensure_manifest_defaults, load_manifest


def _extract_head(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        head = labels.get("head")
        if isinstance(head, str) and head.strip():
            return head.strip()
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def _extract_label(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        for key in ("uid_digit", "agent_icon_id", "amplifier_id", "disc_set_id", "label"):
            value = labels.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("label", "agentId"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build class-balance sampling plan from OCR dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--target-uid-digit", type=int, default=20000)
    parser.add_argument("--target-agent-icon", type=int, default=15000)
    parser.add_argument("--target-equipment", type=int, default=15000)
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    head_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for record in records:
        if not isinstance(record, dict):
            continue
        head = _extract_head(record)
        head_counts[head] += 1
        label_counts[_extract_label(record)] += 1
        locale_counts[str(record.get("locale") or "unknown")] += 1
        resolution_counts[str(record.get("resolution") or "unknown")] += 1
        source_counts[str(record.get("sourceId") or "src_unknown")] += 1

    deficits = {
        "uid_digit": max(0, max(1, args.target_uid_digit) - head_counts.get("uid_digit", 0)),
        "agent_icon": max(0, max(1, args.target_agent_icon) - head_counts.get("agent_icon", 0)),
        "equipment": max(0, max(1, args.target_equipment) - head_counts.get("equipment", 0)),
    }

    per_locale_resolution = defaultdict(int)
    for record in records:
        if not isinstance(record, dict):
            continue
        key = f"{str(record.get('locale') or 'unknown')}:{str(record.get('resolution') or 'unknown')}"
        per_locale_resolution[key] += 1

    plan = {
        "recordCount": len(records),
        "headCounts": dict(head_counts),
        "labelCounts": dict(label_counts),
        "localeCounts": dict(locale_counts),
        "resolutionCounts": dict(resolution_counts),
        "sourceCounts": dict(source_counts),
        "localeResolutionCounts": dict(per_locale_resolution),
        "deficits": deficits,
    }

    if args.output_file:
        output_path = Path(args.output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=True, indent=2)
            fh.write("\n")

    print(json.dumps(plan, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

