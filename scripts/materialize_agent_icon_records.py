from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, hash_file_sha256, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, AGENT_DETAIL_ROLE, EQUIPMENT_ROLE, filter_records, record_screen_role, source_index_from_manifest
from roster_taxonomy import canonicalize_agent_label


# Stable portrait regions on the left side of account-import screens.
_CROP_BOXES = {
    AGENT_DETAIL_ROLE: {
        "portrait_wide": (0.02, 0.08, 0.46, 0.84),
        "portrait_tight": (0.10, 0.10, 0.32, 0.62),
    },
    EQUIPMENT_ROLE: {
        "portrait_wide": (0.02, 0.06, 0.48, 0.88),
        "portrait_tight": (0.08, 0.08, 0.34, 0.66),
    },
}


def _load_image(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _write_image(path: Path, image: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() if path.suffix else ".png"
    extension = ".jpg" if suffix in {".jpg", ".jpeg"} else ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


def _fractional_crop(image: Any, box: tuple[float, float, float, float]) -> Any:
    height, width = image.shape[:2]
    x, y, w, h = box
    x0 = max(0, min(int(round(width * x)), width - 1))
    y0 = max(0, min(int(round(height * y)), height - 1))
    x1 = max(x0 + 1, min(int(round(width * (x + w))), width))
    y1 = max(y0 + 1, min(int(round(height * (y + h))), height))
    crop = image[y0:y1, x0:x1]
    return crop if crop.size else None


def _reviewed_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return {}
    if str(record.get("qaStatus") or "").strip().lower() != "reviewed":
        return {}
    if not isinstance(labels.get("reviewFinal"), dict):
        return {}
    return labels


def _is_derived_agent_icon(record: Dict[str, Any]) -> bool:
    kind = str(record.get("kind") or "").strip()
    if kind == "derived_agent_icon_crop":
        return True
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return False
    derived_from = str(labels.get("derivedFromRecordId") or record.get("derivedFromRecordId") or "").strip()
    return bool(derived_from and kind.startswith("derived_"))


def _resolve_output_root(manifest: Dict[str, Any], explicit_output_root: str) -> Path:
    if str(explicit_output_root).strip():
        return Path(explicit_output_root).resolve()
    private_root = str(manifest.get("privateStorageRoot") or "").strip()
    if private_root:
        return (Path(private_root) / "derived" / "agent_icon").resolve()
    return (Path.cwd() / "dataset_private" / "derived" / "agent_icon").resolve()


def _normalize_agent_icon_label(labels: Dict[str, Any]) -> str:
    for key in ("agent_icon_id", "equipment_agent_id", "agentId", "label"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            canonical = canonicalize_agent_label(value.strip())
            if canonical:
                return canonical
    return ""


def _existing_derived_parents(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("kind") or "").strip() != "derived_agent_icon_crop":
            continue
        labels = record.get("labels")
        if not isinstance(labels, dict):
            continue
        parent_id = str(labels.get("derivedFromRecordId") or record.get("derivedFromRecordId") or "").strip()
        if not parent_id:
            continue
        output.setdefault(parent_id, []).append(str(record.get("id") or ""))
    return output


def _variant_output_path(root: Path, parent_record_id: str, variant_name: str) -> Path:
    safe_parent = parent_record_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    safe_variant = variant_name.replace(":", "_").replace("/", "_").replace("\\", "_")
    return root / safe_parent / f"{safe_parent}__{safe_variant}.png"


def _derived_record(
    *,
    parent: Dict[str, Any],
    parent_role: str,
    agent_icon_id: str,
    variant_name: str,
    crop_box: tuple[float, float, float, float],
    output_path: Path,
    crop_image: Any,
    reviewer_id: str,
    dry_run: bool,
) -> Dict[str, Any]:
    review_final = _reviewed_labels(parent).get("reviewFinal")
    if not isinstance(review_final, dict):
        review_final = {}

    record_id = f"{str(parent.get('id') or 'record')}__agent_icon_{variant_name}"
    capture_hints = parent.get("captureHints")
    capture_hints = dict(capture_hints) if isinstance(capture_hints, dict) else {}
    capture_hints.update(
        {
            "derivedFromRecordId": str(parent.get("id") or ""),
            "derivation": "reviewed_agent_icon_crop",
            "cropVariant": variant_name,
            "cropBox": {
                "x": crop_box[0],
                "y": crop_box[1],
                "width": crop_box[2],
                "height": crop_box[3],
            },
        }
    )

    labels = {
        "head": "agent_icon",
        "agent_icon_id": agent_icon_id,
        "label": agent_icon_id,
        "derivedFromRecordId": str(parent.get("id") or ""),
        "reviewFinal": {
            "reviewer": reviewer_id,
            "reviewedAt": utc_now(),
            "notes": f"Materialized {variant_name} portrait crop from reviewed {parent_role} screen.",
        },
    }
    return {
        "id": record_id,
        "sourceId": str(parent.get("sourceId") or ""),
        "sessionId": str(parent.get("sessionId") or ""),
        "matchId": str(parent.get("matchId") or ""),
        "kind": "derived_agent_icon_crop",
        "head": "agent_icon",
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
            "width": int(crop_image.shape[1]),
            "height": int(crop_image.shape[0]),
        },
        "captureHints": capture_hints,
        "workflow": str(parent.get("workflow") or ACCOUNT_IMPORT_WORKFLOW),
        "screenRole": parent_role,
        "eligibleHeads": ["agent_icon"],
        "ocrImportEligible": True,
        "derivedFromRecordId": str(parent.get("id") or ""),
        "derivedFromPath": str(parent.get("path") or ""),
        "generatedAt": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize reviewed account-import screens into derived agent_icon portrait crops.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true", default=True)
    parser.add_argument("--reviewer-id", default="materialized_agent_icon_crops")
    parser.add_argument("--output-json", default="docs/materialize_agent_icon_records.json")
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

    existing_derived = _existing_derived_parents(records)
    parent_ids_to_refresh: set[str] = set()
    new_records: List[Dict[str, Any]] = []
    processed_parents = 0
    materialized_records = 0
    skipped_missing_assets = 0
    skipped_invalid_label = 0
    skipped_invalid_role = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        if _is_derived_agent_icon(record):
            continue
        parent_role = record_screen_role(record, source_index)
        if parent_role in _CROP_BOXES:
            parent_record_id = str(record.get("id") or "").strip()
            if parent_record_id:
                parent_ids_to_refresh.add(parent_record_id)

    for record in scoped_records:
        if not isinstance(record, dict) or _is_derived_agent_icon(record):
            continue
        labels = _reviewed_labels(record)
        if not labels:
            continue
        parent_role = record_screen_role(record, source_index)
        crop_variants = _CROP_BOXES.get(parent_role)
        if not crop_variants:
            skipped_invalid_role += 1
            continue

        agent_icon_id = _normalize_agent_icon_label(labels)
        if not agent_icon_id:
            skipped_invalid_label += 1
            continue

        record_path = Path(str(record.get("path") or ""))
        image = _load_image(record_path)
        if image is None:
            skipped_missing_assets += 1
            continue

        parent_record_id = str(record.get("id") or "").strip()
        if not parent_record_id:
            continue
        processed_parents += 1

        for variant_name, crop_box in crop_variants.items():
            crop_image = _fractional_crop(image, crop_box)
            if crop_image is None:
                continue
            output_path = _variant_output_path(output_root, parent_record_id, variant_name)
            if not args.dry_run and not _write_image(output_path, crop_image):
                raise RuntimeError(f"Failed to write derived agent icon crop: {output_path}")
            new_records.append(
                _derived_record(
                    parent=record,
                    parent_role=parent_role,
                    agent_icon_id=agent_icon_id,
                    variant_name=variant_name,
                    crop_box=crop_box,
                    output_path=output_path,
                    crop_image=crop_image,
                    reviewer_id=str(args.reviewer_id),
                    dry_run=bool(args.dry_run),
                )
            )
            materialized_records += 1

    if not args.dry_run and parent_ids_to_refresh:
        preserved: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                preserved.append(record)
                continue
            labels = record.get("labels")
            derived_from = ""
            if isinstance(labels, dict):
                derived_from = str(labels.get("derivedFromRecordId") or record.get("derivedFromRecordId") or "").strip()
            if derived_from and derived_from in parent_ids_to_refresh and str(record.get("kind") or "").strip() == "derived_agent_icon_crop":
                continue
            preserved.append(record)
        preserved.extend(new_records)
        manifest["records"] = preserved
        save_manifest(manifest_path, manifest)

    summary = {
        "generatedAt": utc_now(),
        "workflow": str(args.workflow),
        "processedParents": processed_parents,
        "materializedRecords": materialized_records,
        "replacedParentCount": len(parent_ids_to_refresh),
        "existingDerivedParentCount": len(existing_derived),
        "skippedMissingAssets": skipped_missing_assets,
        "skippedInvalidLabel": skipped_invalid_label,
        "skippedInvalidRole": skipped_invalid_role,
        "outputRoot": str(output_root),
        "dryRun": bool(args.dry_run),
    }

    output_json = Path(str(args.output_json or "")).resolve() if str(args.output_json or "").strip() else None
    if output_json and not args.dry_run:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
