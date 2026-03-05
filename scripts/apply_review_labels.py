from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import canonicalize_agent_label


def _normalize_uid_digit(raw: str, fallback: str) -> str:
    candidate = str(raw or "").strip() or str(fallback or "").strip()
    if not candidate:
        return ""
    if len(candidate) == 1 and candidate.isdigit():
        return candidate
    raise ValueError(f"Invalid uid_digit value: {candidate}")


def _normalize_agent_icon(raw: str, fallback: str) -> str:
    candidate = str(raw or "").strip() or str(fallback or "").strip()
    if not candidate:
        return ""
    if candidate == "unknown":
        return "unknown"
    canonical = canonicalize_agent_label(candidate)
    if canonical:
        return canonical
    raise ValueError(f"Invalid agent_icon_id value: {candidate}")


def _normalize_free_text(raw: str, fallback: str) -> str:
    candidate = str(raw or "").strip() or str(fallback or "").strip()
    return candidate


def _normalize_disc_level(raw: str, fallback: str) -> str:
    candidate = str(raw or "").strip() or str(fallback or "").strip()
    if not candidate:
        return ""
    if candidate == "unknown":
        return candidate
    if candidate.isdigit():
        return candidate
    raise ValueError(f"Invalid disc_level value: {candidate}")


def _detect_head(record: Dict[str, Any], labels: Dict[str, Any], suggested: Dict[str, Any]) -> str:
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for payload in (labels, suggested):
        maybe = payload.get("head")
        if isinstance(maybe, str) and maybe.strip():
            return maybe.strip()
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed OCR labels from CSV to dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--review-round", choices=["A", "B", "final"], default="final")
    parser.add_argument("--reviewer-id", default="human")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    record_map = {str(record.get("id") or ""): record for record in records if isinstance(record, dict)}

    applied = 0
    missing = 0
    invalid = 0
    with Path(args.input_csv).resolve().open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record_id = str(row.get("record_id") or "").strip()
            if not record_id:
                continue
            record = record_map.get(record_id)
            if record is None:
                missing += 1
                continue

            labels = record.get("labels")
            if not isinstance(labels, dict):
                labels = {}
                record["labels"] = labels
            suggested = record.get("suggestedLabels")
            if not isinstance(suggested, dict):
                suggested = {}

            review_payload: Dict[str, Any] = {
                "uid_digit": str(row.get("uid_digit") or "").strip(),
                "agent_icon_id": str(row.get("agent_icon_id") or "").strip(),
                "amplifier_id": str(row.get("amplifier_id") or "").strip(),
                "disc_set_id": str(row.get("disc_set_id") or "").strip(),
                "disc_level": str(row.get("disc_level") or "").strip(),
                "reviewer": str(row.get("reviewer") or args.reviewer_id).strip() or args.reviewer_id,
                "reviewedAt": utc_now(),
                "notes": str(row.get("notes") or "").strip(),
            }

            if args.review_round == "A":
                labels["reviewerA"] = review_payload
                record["qaStatus"] = "needs_review"
            elif args.review_round == "B":
                labels["reviewerB"] = review_payload
                record["qaStatus"] = "needs_review"
            else:
                head = _detect_head(record, labels, suggested)
                try:
                    if head == "uid_digit":
                        labels["uid_digit"] = _normalize_uid_digit(review_payload.get("uid_digit", ""), labels.get("uid_digit", "") or suggested.get("uid_digit", ""))
                    elif head == "agent_icon":
                        labels["agent_icon_id"] = _normalize_agent_icon(review_payload.get("agent_icon_id", ""), labels.get("agent_icon_id", "") or suggested.get("agent_icon_id", ""))
                    elif head == "equipment":
                        labels["amplifier_id"] = _normalize_free_text(review_payload.get("amplifier_id", ""), labels.get("amplifier_id", "") or suggested.get("amplifier_id", ""))
                        labels["disc_set_id"] = _normalize_free_text(review_payload.get("disc_set_id", ""), labels.get("disc_set_id", "") or suggested.get("disc_set_id", ""))
                        labels["disc_level"] = _normalize_disc_level(review_payload.get("disc_level", ""), labels.get("disc_level", "") or suggested.get("disc_level", ""))
                    else:
                        labels["uid_digit"] = _normalize_uid_digit(review_payload.get("uid_digit", ""), labels.get("uid_digit", "") or suggested.get("uid_digit", ""))
                except ValueError:
                    invalid += 1
                    record["qaStatus"] = "needs_review"
                    continue

                labels["head"] = head
                labels["label"] = (
                    labels.get("uid_digit")
                    or labels.get("agent_icon_id")
                    or labels.get("amplifier_id")
                    or labels.get("disc_set_id")
                    or labels.get("label")
                    or "unknown"
                )
                labels["reviewFinal"] = {
                    "reviewer": review_payload["reviewer"],
                    "reviewedAt": review_payload["reviewedAt"],
                    "notes": review_payload["notes"],
                }
                record["qaStatus"] = "reviewed"
            applied += 1

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["humanReview"] = "in_progress" if args.review_round in {"A", "B"} else "completed"
        qa_status["qaPass2"] = "completed" if args.review_round == "final" else "pending"
        qa_status["qaUpdatedAt"] = utc_now()

    save_manifest(manifest_path, manifest)
    print(f"Applied labels: {applied}; missing records: {missing}; invalid rows: {invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
