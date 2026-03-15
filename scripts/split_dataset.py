from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ANY_WORKFLOW, filter_records, source_index_from_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_ratio(raw: str) -> tuple[float, float, float]:
    parts = [float(value.strip()) for value in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("ratio must contain exactly three comma-separated numbers.")
    total = sum(parts)
    if total <= 0:
        raise ValueError("ratio sum must be positive.")
    normalized = tuple(value / total for value in parts)
    return normalized  # type: ignore[return-value]


def _group_key(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    record_kind = str(record.get("kind") or "").strip()
    screen_role = str((record.get("source") or {}).get("screenRole") or record.get("screenRole") or "").strip()

    # Keep the same full UID in a single split, including materialized digit samples.
    if record_kind == "derived_uid_digit" or screen_role == "uid_panel":
        uid_value = labels.get("uid_full")
        if isinstance(uid_value, str):
            uid_digits = "".join(ch for ch in uid_value if ch.isdigit())
            if len(uid_digits) == 10:
                return f"uid_full::{uid_digits}"
        uid_payload = labels.get("uid")
        if isinstance(uid_payload, dict):
            uid_value = uid_payload.get("value")
            if isinstance(uid_value, str):
                uid_digits = "".join(ch for ch in uid_value if ch.isdigit())
                if len(uid_digits) == 10:
                    return f"uid_full::{uid_digits}"

    source_id = str(record.get("sourceId") or "src_unknown")
    session_id = str(record.get("sessionId") or "")
    match_id = str(record.get("matchId") or "")
    if session_id:
        return f"{source_id}::session::{session_id}"
    if match_id:
        return f"{source_id}::match::{match_id}"
    record_id = str(record.get("id") or "")
    return f"{source_id}::record::{record_id}"


def assign_splits(records: List[Dict[str, Any]], seed: int, ratio: tuple[float, float, float]) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for idx, record in enumerate(records):
        record_id = str(record.get("id", f"record_{idx}"))
        grouped[_group_key(record)].append(record_id)

    group_ids = list(grouped.keys())
    rng.shuffle(group_ids)
    train_cut = int(len(group_ids) * ratio[0])
    val_cut = train_cut + int(len(group_ids) * ratio[1])
    train_groups = set(group_ids[:train_cut])
    val_groups = set(group_ids[train_cut:val_cut])

    split_ids = {"train": [], "val": [], "test": []}
    for group_id, record_ids in grouped.items():
        if group_id in train_groups:
            split_ids["train"].extend(record_ids)
        elif group_id in val_groups:
            split_ids["val"].extend(record_ids)
        else:
            split_ids["test"].extend(record_ids)
    return {
        "train": split_ids["train"],
        "val": split_ids["val"],
        "test": split_ids["test"],
    }


def _reviewed_truth(record: Dict[str, Any]) -> bool:
    labels = record.get("labels")
    return (
        isinstance(labels, dict)
        and str(record.get("qaStatus") or "").strip().lower() == "reviewed"
        and isinstance(labels.get("reviewFinal"), dict)
    )


def _selected_records(
    payload: Dict[str, Any],
    *,
    workflow: str,
    import_eligible_only: bool,
    reviewed_only: bool,
) -> List[Dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array.")

    if str(workflow or ANY_WORKFLOW).strip().lower() == ANY_WORKFLOW and not import_eligible_only:
        scoped = [record for record in records if isinstance(record, dict)]
    else:
        source_index = source_index_from_manifest(payload.get("sources", []))
        scoped = filter_records(
            records,
            source_index=source_index,
            workflow=workflow,
            import_eligible_only=bool(import_eligible_only),
        )
    if reviewed_only:
        scoped = [record for record in scoped if _reviewed_truth(record)]
    return scoped


def _merge_splits(
    *,
    existing: Dict[str, Any],
    selected_ids: set[str],
    assigned: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {}
    for split_name in ("train", "val", "test"):
        existing_ids = existing.get(split_name)
        preserved = []
        if isinstance(existing_ids, list):
            preserved = [str(record_id) for record_id in existing_ids if str(record_id) not in selected_ids]
        merged[split_name] = [*preserved, *assigned.get(split_name, [])]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate train/val/test split in dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", default="0.8,0.1,0.1")
    parser.add_argument("--workflow", default=ANY_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true")
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--preserve-existing", action="store_true")
    args = parser.parse_args()

    ratio = parse_ratio(args.ratio)
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    payload = ensure_manifest_defaults(load_manifest(manifest_path))
    records = _selected_records(
        payload,
        workflow=str(args.workflow),
        import_eligible_only=bool(args.import_eligible_only),
        reviewed_only=bool(args.reviewed_only),
    )
    if not records:
        raise ValueError("No records matched the requested split selection.")

    assigned = assign_splits(records, args.seed, ratio)
    selected_ids = {str(record.get("id") or "") for record in records if str(record.get("id") or "")}
    payload["splits"] = (
        _merge_splits(existing=payload.get("splits", {}), selected_ids=selected_ids, assigned=assigned)
        if args.preserve_existing
        else assigned
    )
    payload["splitSeed"] = args.seed
    payload["splitRatio"] = {
        "train": ratio[0],
        "val": ratio[1],
        "test": ratio[2],
    }
    payload["splitSelection"] = {
        "workflow": str(args.workflow),
        "importEligibleOnly": bool(args.import_eligible_only),
        "reviewedOnly": bool(args.reviewed_only),
        "preserveExisting": bool(args.preserve_existing),
        "selectedRecordCount": len(records),
        "updatedAt": _utc_now(),
    }

    save_manifest(manifest_path, payload)

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "selectedRecordCount": len(records),
                "workflow": str(args.workflow),
                "importEligibleOnly": bool(args.import_eligible_only),
                "reviewedOnly": bool(args.reviewed_only),
                "preserveExisting": bool(args.preserve_existing),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
