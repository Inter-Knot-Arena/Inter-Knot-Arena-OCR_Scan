from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

from manifest_lib import ensure_manifest_defaults, hash_file_sha256, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, ROSTER_ROLE, UID_PANEL_ROLE, filter_records, record_screen_role, source_index_from_manifest
from scanner.model_runtime import segment_uid_digits
from scanner.screen_runtime import crop_uid_region


def _reviewed_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return {}
    if str(record.get("qaStatus") or "").strip().lower() != "reviewed":
        return {}
    if not isinstance(labels.get("reviewFinal"), dict):
        return {}
    return labels


def _normalize_uid_full(labels: Dict[str, Any]) -> str:
    for key in ("uid_full", "label"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            digits = "".join(ch for ch in value if ch.isdigit())
            if 6 <= len(digits) <= 12:
                return digits
    payload = labels.get("uid")
    if isinstance(payload, dict):
        value = payload.get("value")
        if isinstance(value, str) and value.strip():
            digits = "".join(ch for ch in value if ch.isdigit())
            if 6 <= len(digits) <= 12:
                return digits
    return ""


def _existing_derived_index(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        labels = record.get("labels")
        if not isinstance(labels, dict):
            continue
        derived_from = str(labels.get("derivedFromRecordId") or record.get("derivedFromRecordId") or "").strip()
        if not derived_from:
            continue
        if str(record.get("kind") or "").strip() != "derived_uid_digit":
            continue
        output.setdefault(derived_from, []).append(record)
    return output


def _resolve_output_root(manifest: Dict[str, Any], explicit_output_root: str) -> Path:
    if str(explicit_output_root).strip():
        return Path(explicit_output_root).resolve()
    private_root = str(manifest.get("privateStorageRoot") or "").strip()
    if private_root:
        return (Path(private_root) / "derived" / "uid_digits").resolve()
    return (Path.cwd() / "dataset_private" / "derived" / "uid_digits").resolve()


def _parent_role(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> str:
    role = record_screen_role(record, source_index)
    return role if role in {UID_PANEL_ROLE, ROSTER_ROLE} else ""


def _digit_output_path(root: Path, parent_record_id: str, digit_index: int, digit_label: str) -> Path:
    safe_parent = parent_record_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    return root / safe_parent / f"{safe_parent}_digit_{digit_index + 1:02d}_{digit_label}.png"


def _write_digit_crop(path: Path, image: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image))


def _derived_record(
    *,
    parent: Dict[str, Any],
    parent_role: str,
    uid_full: str,
    digit_label: str,
    digit_index: int,
    output_path: Path,
    digit_image: Any,
    dry_run: bool,
) -> Dict[str, Any]:
    image = digit_image
    if image is None:
        image = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to load derived digit crop: {output_path}")

    review_final = _reviewed_labels(parent).get("reviewFinal")
    if not isinstance(review_final, dict):
        review_final = {}

    record_id = f"{str(parent.get('id') or 'record')}__uid_digit_{digit_index + 1:02d}"
    capture_hints = parent.get("captureHints")
    capture_hints = dict(capture_hints) if isinstance(capture_hints, dict) else {}
    capture_hints.update(
        {
            "derivedFromRecordId": str(parent.get("id") or ""),
            "derivation": "reviewed_uid_materialized_digit",
            "digitIndex": digit_index,
            "digitLabel": digit_label,
        }
    )
    labels = {
        "uid_digit": digit_label,
        "label": digit_label,
        "head": "uid_digit",
        "digitIndex": digit_index,
        "uid_full": uid_full,
        "derivedFromRecordId": str(parent.get("id") or ""),
        "reviewFinal": {
            "reviewer": str(review_final.get("reviewer") or "materialized_uid_digits"),
            "reviewedAt": utc_now(),
            "notes": f"Materialized digit {digit_index + 1} from reviewed {parent_role} UID.",
        },
    }
    return {
        "id": record_id,
        "sourceId": str(parent.get("sourceId") or ""),
        "sessionId": str(parent.get("sessionId") or ""),
        "matchId": str(parent.get("matchId") or ""),
        "kind": "derived_uid_digit",
        "head": "uid_digit",
        "state": str(parent.get("state") or "other"),
        "locale": str(parent.get("locale") or ""),
        "resolution": str(parent.get("resolution") or ""),
        "path": str(output_path),
        "labelsPath": "",
        "qaStatus": "reviewed",
        "labels": labels,
        "suggestedLabels": {},
        "sha256": "" if dry_run else hash_file_sha256(output_path),
        "originalFileName": output_path.name,
        "dimensions": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "captureHints": capture_hints,
        "workflow": str(parent.get("workflow") or ACCOUNT_IMPORT_WORKFLOW),
        "screenRole": parent_role,
        "eligibleHeads": ["uid_digit"],
        "ocrImportEligible": True,
        "derivedFromRecordId": str(parent.get("id") or ""),
        "derivedFromPath": str(parent.get("path") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize reviewed full-UID screenshots into per-digit OCR samples.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true", default=True)
    parser.add_argument("--allow-roster", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    source_index = source_index_from_manifest(manifest.get("sources", []))
    output_root = _resolve_output_root(manifest, args.output_root)
    scoped_records = filter_records(
        records,
        source_index=source_index,
        workflow=args.workflow,
        import_eligible_only=bool(args.import_eligible_only),
    )
    derived_index = _existing_derived_index(records)

    parent_ids_to_replace: set[str] = set()
    new_records: List[Dict[str, Any]] = []
    processed = 0
    materialized = 0
    skipped_mismatch = 0
    skipped_missing = 0

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        labels = _reviewed_labels(record)
        if not labels:
            continue
        parent_role = _parent_role(record, source_index)
        if parent_role == ROSTER_ROLE and not bool(args.allow_roster):
            continue
        if parent_role not in {UID_PANEL_ROLE, ROSTER_ROLE}:
            continue

        uid_full = _normalize_uid_full(labels)
        if not uid_full:
            continue
        record_path = Path(str(record.get("path") or ""))
        if not record_path.exists():
            skipped_missing += 1
            continue
        image = cv2.imread(str(record_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped_missing += 1
            continue
        uid_crop = crop_uid_region(image, parent_role)
        if uid_crop is None:
            skipped_missing += 1
            continue
        digit_crops = segment_uid_digits(uid_crop)
        if len(digit_crops) != len(uid_full):
            skipped_mismatch += 1
            continue

        parent_record_id = str(record.get("id") or "")
        if not parent_record_id:
            continue
        parent_ids_to_replace.add(parent_record_id)
        processed += 1

        for digit_index, (digit_label, digit_crop) in enumerate(zip(uid_full, digit_crops)):
            output_path = _digit_output_path(output_root, parent_record_id, digit_index, digit_label)
            if not args.dry_run and not _write_digit_crop(output_path, digit_crop):
                raise RuntimeError(f"Failed to write derived digit crop: {output_path}")
            new_records.append(
                _derived_record(
                    parent=record,
                    parent_role=parent_role,
                    uid_full=uid_full,
                    digit_label=digit_label,
                    digit_index=digit_index,
                    output_path=output_path,
                    digit_image=digit_crop,
                    dry_run=bool(args.dry_run),
                )
            )
            materialized += 1

    if not args.dry_run and parent_ids_to_replace:
        preserved: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                preserved.append(record)
                continue
            labels = record.get("labels")
            derived_from = ""
            if isinstance(labels, dict):
                derived_from = str(labels.get("derivedFromRecordId") or record.get("derivedFromRecordId") or "").strip()
            if derived_from and derived_from in parent_ids_to_replace and str(record.get("kind") or "").strip() == "derived_uid_digit":
                continue
            preserved.append(record)
        preserved.extend(new_records)
        manifest["records"] = preserved
        save_manifest(manifest_path, manifest)

    summary = {
        "generatedAt": utc_now(),
        "workflow": str(args.workflow),
        "processedParents": processed,
        "materializedDigits": materialized,
        "replacedParents": sorted(parent_ids_to_replace),
        "existingDerivedParents": sorted(derived_index.keys()),
        "skippedMissingAssets": skipped_missing,
        "skippedSegmentationMismatch": skipped_mismatch,
        "outputRoot": str(output_root),
        "dryRun": bool(args.dry_run),
    }
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
