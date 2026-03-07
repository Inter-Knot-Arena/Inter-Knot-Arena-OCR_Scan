from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, filter_records, record_screen_role, record_workflow, source_index_from_manifest
from roster_taxonomy import current_agent_ids, source_focus_agent_ids


FIELDS = [
    "batch_id",
    "priority_score",
    "priority_reasons",
    "record_id",
    "qa_status",
    "source_id",
    "focus_agent_id",
    "workflow",
    "screen_role",
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
    value = record.get(key)
    if not isinstance(value, dict):
        return {}
    return value


def _str_value(payload: Dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return str(value).strip() if isinstance(value, str) else ""


def _confidence_value(payload: Dict[str, Any]) -> float:
    value = payload.get("confidence")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _detect_head(record: Dict[str, Any]) -> str:
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    suggested = _labels(record, "suggestedLabels")
    head = _str_value(suggested, "head")
    return head or "unknown"


def _reviewed_agent_counts(records: List[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        labels = _labels(record, "labels")
        value = _str_value(labels, "agent_icon_id")
        if value and value != "unknown":
            counts[value] += 1
    return counts


def _focus_agent(source_index: Dict[str, Dict[str, Any]], source_id: str) -> str:
    source = source_index.get(source_id, {})
    focus_ids = source_focus_agent_ids(source)
    return focus_ids[0] if focus_ids else ""


def _score_record(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]], missing_reviewed: set[str]) -> tuple[int, List[str], str]:
    head = _detect_head(record)
    score = 0
    reasons: List[str] = []
    source_id = str(record.get("sourceId") or "src_unknown")
    focus_agent_id = _focus_agent(source_index, source_id)
    suggested = _labels(record, "suggestedLabels")
    suggested_agent = _str_value(suggested, "agent_icon_id")
    status = str(record.get("qaStatus") or "").strip().lower()

    if status == "needs_review":
        score += 18
        reasons.append("needs_review")
    elif status == "unlabeled":
        score += 8
        reasons.append("unlabeled")

    if head == "uid_digit":
        score += 42
        reasons.append("uid_digit")
    elif head == "agent_icon":
        score += 26
        reasons.append("agent_icon")
    elif head == "equipment":
        score += 12
        reasons.append("equipment")

    if focus_agent_id and focus_agent_id in missing_reviewed:
        score += 60
        reasons.append("missing_reviewed_focus")
    if suggested_agent and suggested_agent in missing_reviewed:
        score += 50
        reasons.append("missing_reviewed_suggestion")
    if head == "agent_icon" and not suggested_agent:
        score += 25
        reasons.append("missing_agent_suggestion")

    confidence = _confidence_value(suggested)
    if confidence < 0.45:
        score += 22
        reasons.append("very_low_conf")
    elif confidence < 0.60:
        score += 12
        reasons.append("low_conf")

    group_key = head
    if focus_agent_id:
        group_key = f"{head}:{focus_agent_id}"
    elif suggested_agent:
        group_key = f"{head}:{suggested_agent}"
    return score, reasons, group_key


def _round_robin_order(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in scored:
        buckets[str(item["groupKey"])].append(item)
    for values in buckets.values():
        values.sort(key=lambda row: (-int(row["priorityScore"]), str(row["record"]["id"])))

    ordered: List[Dict[str, Any]] = []
    while buckets:
        keys = sorted(buckets.keys(), key=lambda key: (-int(buckets[key][0]["priorityScore"]), key))
        for key in keys:
            ordered.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    return ordered


def _row(item: Dict[str, Any], source_index: Dict[str, Dict[str, Any]], batch_id: str) -> Dict[str, str]:
    record = item["record"]
    reviewed = _labels(record, "labels")
    suggested = _labels(record, "suggestedLabels")
    source_id = str(record.get("sourceId") or "src_unknown")
    return {
        "batch_id": batch_id,
        "priority_score": str(item["priorityScore"]),
        "priority_reasons": ";".join(item["priorityReasons"]),
        "record_id": str(record.get("id") or ""),
        "qa_status": str(record.get("qaStatus") or ""),
        "source_id": source_id,
        "focus_agent_id": _focus_agent(source_index, source_id),
        "workflow": record_workflow(record, source_index),
        "screen_role": record_screen_role(record, source_index),
        "head": _detect_head(record),
        "locale": str(record.get("locale") or ""),
        "resolution": str(record.get("resolution") or ""),
        "path": str(record.get("path") or ""),
        "reviewed_uid_digit": _str_value(reviewed, "uid_digit"),
        "suggested_uid_digit": _str_value(suggested, "uid_digit"),
        "reviewed_agent_icon_id": _str_value(reviewed, "agent_icon_id"),
        "suggested_agent_icon_id": _str_value(suggested, "agent_icon_id"),
        "reviewed_amplifier_id": _str_value(reviewed, "amplifier_id"),
        "suggested_amplifier_id": _str_value(suggested, "amplifier_id"),
        "reviewed_disc_set_id": _str_value(reviewed, "disc_set_id"),
        "suggested_disc_set_id": _str_value(suggested, "disc_set_id"),
        "reviewed_disc_level": _str_value(reviewed, "disc_level"),
        "suggested_disc_level": _str_value(suggested, "disc_level"),
        "suggested_confidence": f"{_confidence_value(suggested):.4f}" if suggested else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build prioritized OCR human-review batches.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-csv", default="docs/review_queue.priority.csv")
    parser.add_argument("--output-json", default="docs/review_batches.json")
    parser.add_argument("--status", default="needs_review")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--max-batches", type=int, default=10)
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

    reviewed_counts = _reviewed_agent_counts(scoped_records)
    missing_reviewed = {agent for agent in current_agent_ids() if reviewed_counts.get(agent, 0) <= 0}
    status_filter = {item.strip().lower() for item in str(args.status).split(",") if item.strip()}

    scored: List[Dict[str, Any]] = []
    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        status = str(record.get("qaStatus") or "").strip().lower()
        if status_filter and status not in status_filter:
            continue
        priority_score, priority_reasons, group_key = _score_record(record, source_index, missing_reviewed)
        if priority_score <= 0:
            continue
        scored.append(
            {
                "record": record,
                "priorityScore": priority_score,
                "priorityReasons": priority_reasons,
                "groupKey": group_key,
            }
        )

    ordered = _round_robin_order(scored)
    max_rows = max(1, int(args.batch_size)) * max(1, int(args.max_batches))
    ordered = ordered[:max_rows]

    rows: List[Dict[str, str]] = []
    batches: List[Dict[str, Any]] = []
    for index in range(0, len(ordered), max(1, int(args.batch_size))):
        chunk = ordered[index : index + max(1, int(args.batch_size))]
        batch_id = f"ocr-review-{(index // max(1, int(args.batch_size))) + 1:03d}"
        head_counts = Counter()
        focus_counts = Counter()
        for item in chunk:
            record = item["record"]
            head_counts[_detect_head(record)] += 1
            source_id = str(record.get("sourceId") or "src_unknown")
            focus_agent_id = _focus_agent(source_index, source_id)
            if focus_agent_id:
                focus_counts[focus_agent_id] += 1
            rows.append(_row(item, source_index, batch_id))
        batches.append(
            {
                "batchId": batch_id,
                "size": len(chunk),
                "maxPriority": max(int(item["priorityScore"]) for item in chunk),
                "headCounts": dict(head_counts),
                "focusAgentCounts": dict(focus_counts),
            }
        )

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generatedAt": utc_now(),
        "workflow": str(args.workflow),
        "importEligibleOnly": bool(args.import_eligible_only),
        "statusFilter": sorted(status_filter),
        "batchSize": max(1, int(args.batch_size)),
        "batchCount": len(batches),
        "queuedRecords": len(rows),
        "missingReviewedAgentIcons": sorted(missing_reviewed),
        "batches": batches,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
