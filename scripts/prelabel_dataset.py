from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, filter_records, source_index_from_manifest
from roster_taxonomy import canonicalize_agent_label
from scanner.model_runtime import (
    ModelRegistry,
    preprocess_digit_for_classifier,
    preprocess_icon_for_classifier,
)


def _has_labels(record: Dict[str, Any]) -> bool:
    labels = record.get("labels")
    if isinstance(labels, dict):
        if isinstance(labels.get("reviewFinal"), dict):
            return True
        if any(isinstance(value, str) and value for value in labels.values()):
            return True
    suggested = record.get("suggestedLabels")
    return isinstance(suggested, dict) and any(isinstance(value, str) and value for value in suggested.values())


def _detect_head(record: Dict[str, Any]) -> str:
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    labels = record.get("labels")
    if isinstance(labels, dict):
        maybe = labels.get("head")
        if isinstance(maybe, str) and maybe.strip():
            return maybe.strip()
    return "uid_digit"


def _center_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, (w - width) // 2)
    y0 = max(0, (h - height) // 2)
    x1 = min(w, x0 + width)
    y1 = min(h, y0 + height)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return crop


def _predict_uid_digit(image: np.ndarray) -> Tuple[str, float]:
    classifier = ModelRegistry.uid_classifier()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = _center_crop(gray, 16, 24)
    pred = classifier.predict(preprocess_digit_for_classifier(crop, classifier))
    return pred.label, float(pred.confidence)


def _predict_agent_icon(image: np.ndarray) -> Tuple[str, float]:
    classifier = ModelRegistry.agent_classifier()
    prediction = classifier.predict(preprocess_icon_for_classifier(image, classifier))
    return prediction.label, float(prediction.confidence)


def _prelabel_record(record: Dict[str, Any], threshold: float) -> Tuple[bool, str]:
    path_value = str(record.get("path") or "")
    if not path_value:
        return False, "missing_path"
    image_path = Path(path_value)
    if not image_path.exists():
        return False, "path_not_found"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False, "decode_failed"

    head = _detect_head(record)
    labels: Dict[str, Any] = {
        "head": head,
        "prelabelVersion": "ocr-prelabel-v1",
        "prelabelAt": utc_now(),
    }

    confidence = 0.0
    if head == "uid_digit":
        if not ModelRegistry.has_uid_model():
            return False, "uid_model_missing"
        uid_digit, confidence = _predict_uid_digit(image)
        labels["uid_digit"] = uid_digit
        labels["label"] = uid_digit
    elif head == "agent_icon":
        if not ModelRegistry.has_agent_model():
            return False, "agent_model_missing"
        agent_id, confidence = _predict_agent_icon(image)
        canonical = canonicalize_agent_label(agent_id)
        labels["agent_icon_id"] = canonical or "unknown"
        labels["label"] = canonical or "unknown"
    else:
        labels["label"] = "unknown"
        labels["amplifier_id"] = "unknown"
        labels["disc_set_id"] = "unknown"
        labels["disc_level"] = "unknown"
        confidence = 0.0

    labels["confidence"] = round(confidence, 4)
    record["suggestedLabels"] = labels
    record["qaStatus"] = "needs_review"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Semi-automatic prelabel for OCR dataset records.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true", default=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = source_index_from_manifest(manifest.get("sources", []))
    candidate_records = filter_records(
        records,
        source_index=source_index,
        workflow=args.workflow,
        import_eligible_only=bool(args.import_eligible_only),
    )

    processed = 0
    updated = 0
    skipped_existing = 0
    errors: Dict[str, int] = {}
    limit = max(0, int(args.max_records))
    threshold = max(0.0, min(1.0, args.confidence_threshold))

    for record in candidate_records:
        if not isinstance(record, dict):
            continue
        if not args.overwrite and _has_labels(record):
            skipped_existing += 1
            continue
        if limit and processed >= limit:
            break
        processed += 1
        ok, reason = _prelabel_record(record=record, threshold=threshold)
        if ok:
            updated += 1
        else:
            errors[reason] = errors.get(reason, 0) + 1

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["prelabel"] = "completed" if updated > 0 else "pending"
        qa_status["prelabelUpdatedAt"] = utc_now()
        qa_status["prelabelProcessed"] = processed
        qa_status["prelabelUpdated"] = updated

    save_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "processed": processed,
                "updated": updated,
                "skippedExisting": skipped_existing,
                "errors": errors,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
