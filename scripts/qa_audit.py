from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now
from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, filter_records, source_index_from_manifest


def _label_payload(record: Dict[str, Any], suggested: bool) -> Dict[str, Any]:
    key = "suggestedLabels" if suggested else "labels"
    labels = record.get(key)
    if not isinstance(labels, dict):
        return {}
    return labels


def _detect_head(record: Dict[str, Any], *, suggested: bool) -> str:
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    labels = _label_payload(record, suggested=suggested)
    if labels:
        maybe = labels.get("head")
        if isinstance(maybe, str) and maybe.strip():
            return maybe.strip()
    return "unknown"


def _extract_label(record: Dict[str, Any], *, suggested: bool) -> str:
    labels = _label_payload(record, suggested=suggested)
    head = _detect_head(record, suggested=suggested)
    if not labels:
        return ""
    for key in ("uid_digit", "agent_icon_id", "amplifier_id", "disc_set_id"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if head != "agent_icon":
        value = labels.get("label")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_owned_agent_ids(record: Dict[str, Any], *, suggested: bool) -> List[str]:
    labels = _label_payload(record, suggested=suggested)
    values = labels.get("owned_agent_ids")
    if not isinstance(values, list):
        return []
    output: List[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            output.append(value.strip())
    return output


def _sample_for_double_review(records: List[Dict[str, Any]], ratio: float, seed: int) -> List[str]:
    candidates = [str(record.get("id") or "") for record in records if isinstance(record, dict)]
    candidates = [item for item in candidates if item]
    if not candidates:
        return []
    rng = random.Random(seed)
    rng.shuffle(candidates)
    take = max(1, int(len(candidates) * max(0.0, min(1.0, ratio))))
    return candidates[:take]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate QA audit report for OCR manifest labels.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-file", default="docs/qa_report.json")
    parser.add_argument("--double-review-file", default="docs/double_review_samples.json")
    parser.add_argument("--double-review-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true", default=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = source_index_from_manifest(manifest.get("sources", []))
    scoped_records = filter_records(
        records,
        source_index=source_index,
        workflow=args.workflow,
        import_eligible_only=bool(args.import_eligible_only),
    )

    valid_records: List[Dict[str, Any]] = []
    reviewed = 0
    reviewed_records = 0
    suggested = 0
    needs_review = 0
    head_counts: Dict[str, int] = {}
    suggested_head_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    suggested_label_counts: Dict[str, int] = {}
    roster_reviewed_count = 0
    roster_owned_agents: set[str] = set()
    roster_not_owned_agents: set[str] = set()
    low_conf_count = 0
    suggested_low_conf_count = 0

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        valid_records.append(record)
        head = _detect_head(record, suggested=False)
        head_counts[head] = head_counts.get(head, 0) + 1
        suggested_head = _detect_head(record, suggested=True)
        suggested_head_counts[suggested_head] = suggested_head_counts.get(suggested_head, 0) + 1

        label = _extract_label(record, suggested=False)
        reviewed_owned = _extract_owned_agent_ids(record, suggested=False)
        if label:
            reviewed += 1
            label_counts[label] = label_counts.get(label, 0) + 1
        if label or reviewed_owned or str(record.get("qaStatus") or "").lower() == "reviewed":
            reviewed_records += 1
        if str(record.get("screenRole") or "").strip().lower() == "roster" and reviewed_owned:
            roster_reviewed_count += 1
            roster_owned_agents.update(reviewed_owned)
            not_owned_values = _label_payload(record, suggested=False).get("not_owned_agent_ids")
            if isinstance(not_owned_values, list):
                roster_not_owned_agents.update(str(item).strip() for item in not_owned_values if isinstance(item, str) and str(item).strip())
        suggested_label = _extract_label(record, suggested=True)
        if suggested_label:
            suggested += 1
            suggested_label_counts[suggested_label] = suggested_label_counts.get(suggested_label, 0) + 1

        if str(record.get("qaStatus") or "").lower() == "needs_review":
            needs_review += 1

        labels = _label_payload(record, suggested=False)
        if labels:
            confidence_raw = labels.get("confidence")
            if isinstance(confidence_raw, (int, float)) and float(confidence_raw) < 0.6:
                low_conf_count += 1
        suggested_labels = _label_payload(record, suggested=True)
        if suggested_labels:
            confidence_raw = suggested_labels.get("confidence")
            if isinstance(confidence_raw, (int, float)) and float(confidence_raw) < 0.6:
                suggested_low_conf_count += 1

    report = {
        "recordCount": len(valid_records),
        "workflow": str(args.workflow),
        "importEligibleOnly": bool(args.import_eligible_only),
        "labeledCount": reviewed,
        "reviewedCount": reviewed_records,
        "reviewedScalarLabelCount": reviewed,
        "suggestedCount": suggested,
        "needsReviewCount": needs_review,
        "reviewedRosterRecordCount": roster_reviewed_count,
        "reviewedRosterOwnedAgentCount": len(roster_owned_agents),
        "reviewedRosterNotOwnedAgentCount": len(roster_not_owned_agents),
        "reviewedRosterOwnedAgentIds": sorted(roster_owned_agents),
        "reviewedRosterNotOwnedAgentIds": sorted(roster_not_owned_agents),
        "lowConfidenceCount": low_conf_count,
        "suggestedLowConfidenceCount": suggested_low_conf_count,
        "headCounts": head_counts,
        "suggestedHeadCounts": suggested_head_counts,
        "labelCounts": label_counts,
        "suggestedLabelCounts": suggested_label_counts,
        "generatedAt": utc_now(),
    }

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    review_path = Path(args.double_review_file).resolve()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "recordIds": _sample_for_double_review(
                    records=valid_records,
                    ratio=max(0.0, min(1.0, args.double_review_ratio)),
                    seed=args.seed,
                ),
                "generatedAt": utc_now(),
            },
            fh,
            ensure_ascii=True,
            indent=2,
        )
        fh.write("\n")

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["doubleReviewRate"] = max(0.0, min(1.0, args.double_review_ratio))
        qa_status["humanReview"] = "in_progress" if needs_review > 0 else "completed"
        qa_status["qaPass2"] = "pending"
        qa_status["qaUpdatedAt"] = utc_now()

    save_manifest(manifest_path, manifest)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
