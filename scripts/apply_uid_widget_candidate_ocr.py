from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

_UID_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")


def _normalize_uid_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    match = _UID_RE.search(value)
    if match:
        return match.group(1)
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits if len(digits) == 10 else ""


def _load_recovery_report(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery report must be a JSON object")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("recovery report records must be an array")
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        record_id = str(row.get("id") or "").strip()
        if not record_id:
            continue
        output[record_id] = row
    return output


def _load_candidate_ocr(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("candidate OCR payload must be an array")
    output: Dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        output[str(row.get("id") or "").strip()] = _normalize_uid_text(str(row.get("text") or ""))
    return output


def _resolve_uid(values: List[str]) -> tuple[str, str]:
    normalized = [value for value in values if value]
    if not normalized:
        return ("", "no_ocr")
    counts = Counter(normalized)
    if len(counts) == 1:
        return (next(iter(counts.keys())), "single_unique")
    most_common = counts.most_common()
    best_uid, best_count = most_common[0]
    second_count = most_common[1][1] if len(most_common) > 1 else 0
    if best_count >= 3 and second_count <= 1:
        return (best_uid, "strong_majority")
    return ("", "conflicting_ocr")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply adaptive widget OCR truth to unresolved uid_panel records.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--recovery-report", required=True)
    parser.add_argument("--candidate-ocr-json", action="append", required=True)
    parser.add_argument("--reviewer-id", default="adaptive_uid_widget_ocr")
    parser.add_argument("--output-json", default="docs/apply_uid_widget_candidate_ocr.json")
    parser.add_argument("--output-csv", default="docs/apply_uid_widget_candidate_ocr.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    recovery_rows = _load_recovery_report(Path(args.recovery_report).resolve())
    candidate_payloads = [_load_candidate_ocr(Path(path).resolve()) for path in args.candidate_ocr_json]

    by_record: Dict[str, List[str]] = defaultdict(list)
    for payload in candidate_payloads:
        for candidate_id, value in payload.items():
            if not candidate_id or "::" not in candidate_id:
                continue
            record_id, _ = candidate_id.split("::", 1)
            by_record[record_id].append(value)

    applied = 0
    unresolved = 0
    report_rows: List[Dict[str, Any]] = []

    record_index = {
        str(record.get("id") or ""): record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and str(record.get("id") or "")
    }

    for record_id, recovery_row in sorted(recovery_rows.items()):
        if str(recovery_row.get("status") or "") != "unresolved":
            continue
        record = record_index.get(record_id)
        if not isinstance(record, dict):
            continue
        resolved_uid, resolution = _resolve_uid(by_record.get(record_id, []))
        row = {
            "id": record_id,
            "currentUidFull": str(recovery_row.get("currentUidFull") or ""),
            "predictedUidFull": str(recovery_row.get("predictedUidFull") or ""),
            "resolvedUidFull": resolved_uid,
            "resolution": resolution,
        }
        if not resolved_uid:
            unresolved += 1
            report_rows.append(row)
            continue

        labels = record.get("labels")
        labels = labels if isinstance(labels, dict) else {}
        labels["uid_full"] = resolved_uid
        labels["uid"] = {"value": resolved_uid}
        labels["label"] = resolved_uid
        labels["reviewFinal"] = {
            "reviewer": str(args.reviewer_id),
            "reviewedAt": utc_now(),
            "notes": f"Recovered from adaptive UID widget OCR; resolution={resolution}.",
        }
        record["labels"] = labels
        record["qaStatus"] = "reviewed"
        applied += 1
        report_rows.append(row)

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    output_json_path = Path(args.output_json).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": utc_now(),
        "manifest": str(manifest_path),
        "recoveryReport": str(Path(args.recovery_report).resolve()),
        "candidateOcrJsons": [str(Path(path).resolve()) for path in args.candidate_ocr_json],
        "appliedCount": applied,
        "remainingUnresolvedCount": unresolved,
        "rows": report_rows,
        "dryRun": bool(args.dry_run),
    }
    output_json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    output_csv_path = Path(args.output_csv).resolve()
    with output_csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "currentUidFull", "predictedUidFull", "resolvedUidFull", "resolution"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(
        json.dumps(
            {
                "appliedCount": applied,
                "remainingUnresolvedCount": unresolved,
                "outputJson": str(output_json_path),
                "outputCsv": str(output_csv_path),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
