from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from materialize_uid_digit_records import _load_image, _materialize_digit_crops
from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, UID_PANEL_ROLE, filter_records, source_index_from_manifest
from train_ocr_models import _preprocess

_UID_DIGIT_COUNT = 10
_UID_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")


def _normalize_uid_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str):
        return ""
    match = _UID_RE.search(value)
    if match:
        return match.group(1)
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits if len(digits) == _UID_DIGIT_COUNT else ""


def _reviewed_uid_panel_records(manifest: Dict[str, Any], workflow: str) -> List[Dict[str, Any]]:
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = source_index_from_manifest(manifest.get("sources", []))
    scoped = filter_records(records, source_index=source_index, workflow=workflow, import_eligible_only=True)
    output: List[Dict[str, Any]] = []
    for record in scoped:
        if not isinstance(record, dict):
            continue
        role = str((record.get("source") or {}).get("screenRole") or record.get("screenRole") or "").strip()
        if role != UID_PANEL_ROLE:
            continue
        if str(record.get("kind") or "").strip() == "derived_uid_digit":
            continue
        labels = record.get("labels")
        if not isinstance(labels, dict):
            continue
        if str(record.get("qaStatus") or "").strip().lower() != "reviewed":
            continue
        if not isinstance(labels.get("reviewFinal"), dict):
            continue
        output.append(record)
    return output


def _current_uid(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    for key in ("uid_full", "uid", "label"):
        normalized = _normalize_uid_text(labels.get(key))
        if normalized:
            return normalized
    return ""


def _load_widget_ocr(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("widget OCR payload must be a JSON array")
    output: Dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        record_id = str(item.get("id") or "").strip()
        if not record_id:
            continue
        output[record_id] = _normalize_uid_text(item.get("text"))
    return output


def _digit_features(record: Dict[str, Any]) -> List[np.ndarray]:
    path = Path(str(record.get("path") or ""))
    if not path.exists():
        return []
    image = _load_image(path, mode=1)
    if image is None:
        return []
    digit_crops = _materialize_digit_crops(image, UID_PANEL_ROLE, _UID_DIGIT_COUNT)
    if len(digit_crops) != _UID_DIGIT_COUNT:
        return []
    return [_preprocess(np.asarray(crop), "uid_digit") for crop in digit_crops]


def _fit_digit_model(samples: List[Tuple[List[np.ndarray], str]]) -> LogisticRegression:
    x_rows: List[np.ndarray] = []
    y_rows: List[str] = []
    for digit_features, uid_full in samples:
        if len(digit_features) != len(uid_full):
            continue
        for feature, digit_label in zip(digit_features, uid_full):
            x_rows.append(np.asarray(feature, dtype=np.float32))
            y_rows.append(str(digit_label))
    if not x_rows or len(set(y_rows)) < 2:
        raise ValueError("Not enough trusted UID digits to fit the bootstrap classifier.")
    x = np.vstack(x_rows).astype(np.float32)
    y = np.array(y_rows)
    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(x, y)
    return model


def _predict_uid(model: LogisticRegression, digit_features: List[np.ndarray]) -> Tuple[str, float]:
    if not digit_features:
        return ("", 0.0)
    x = np.vstack([np.asarray(feature, dtype=np.float32) for feature in digit_features]).astype(np.float32)
    predictions = [str(item) for item in model.predict(x)]
    probabilities = model.predict_proba(x)
    confidence_floor = float(min(float(np.max(row)) for row in probabilities)) if probabilities.size else 0.0
    return ("".join(predictions), confidence_floor)


def _record_summary(
    record: Dict[str, Any],
    *,
    current_uid: str,
    widget_ocr_uid: str,
    predicted_uid: str,
    min_confidence: float,
    status: str,
    resolved_uid: str = "",
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "id": str(record.get("id") or ""),
        "sessionId": str(record.get("sessionId") or ""),
        "path": str(record.get("path") or ""),
        "currentUidFull": current_uid,
        "widgetOcrUidFull": widget_ocr_uid,
        "predictedUidFull": predicted_uid,
        "predictionMinConfidence": round(float(min_confidence), 6),
        "status": status,
        "resolvedUidFull": resolved_uid,
        "reason": reason,
    }


def _clear_review(record: Dict[str, Any]) -> None:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return
    labels.pop("reviewFinal", None)
    record["qaStatus"] = "needs_review"


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover reviewed uid_panel truth from widget OCR plus bootstrap digit agreement.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--widget-ocr-json", required=True)
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--reviewer-id", default="auto_uid_widget_recovery")
    parser.add_argument("--output-json", default="docs/recover_uid_panel_truth.json")
    parser.add_argument("--output-csv", default="docs/recover_uid_panel_truth.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-unresolved-reviewed", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = _reviewed_uid_panel_records(manifest, workflow=str(args.workflow))
    widget_ocr_index = _load_widget_ocr(Path(args.widget_ocr_json).resolve())

    rows_by_id: Dict[str, Dict[str, Any]] = {}
    trusted_segmentable: Dict[str, str] = {}
    safe_current_only: Dict[str, str] = {}
    features_by_id: Dict[str, List[np.ndarray]] = {}
    predicted_by_id: Dict[str, Tuple[str, float]] = {}

    for record in records:
        record_id = str(record.get("id") or "")
        current_uid = _current_uid(record)
        widget_ocr_uid = widget_ocr_index.get(record_id, "")
        digit_features = _digit_features(record)
        features_by_id[record_id] = digit_features

        if current_uid and widget_ocr_uid and current_uid == widget_ocr_uid:
            safe_current_only[record_id] = current_uid
            if len(digit_features) == _UID_DIGIT_COUNT:
                trusted_segmentable[record_id] = current_uid

        rows_by_id[record_id] = _record_summary(
            record,
            current_uid=current_uid,
            widget_ocr_uid=widget_ocr_uid,
            predicted_uid="",
            min_confidence=0.0,
            status="unresolved",
            reason="pending_bootstrap",
        )

    if len(trusted_segmentable) < 10:
        raise RuntimeError("Too few trustworthy UID panels to bootstrap digit recovery.")

    bootstrap_iterations: List[Dict[str, Any]] = []
    final_model: LogisticRegression | None = None
    while True:
        training_samples: List[Tuple[List[np.ndarray], str]] = []
        for record in records:
            record_id = str(record.get("id") or "")
            trusted_uid = trusted_segmentable.get(record_id)
            digit_features = features_by_id.get(record_id) or []
            if trusted_uid and len(digit_features) == _UID_DIGIT_COUNT:
                training_samples.append((digit_features, trusted_uid))

        model = _fit_digit_model(training_samples)
        final_model = model
        new_ids: List[str] = []
        for record in records:
            record_id = str(record.get("id") or "")
            if record_id in trusted_segmentable:
                continue
            digit_features = features_by_id.get(record_id) or []
            widget_ocr_uid = widget_ocr_index.get(record_id, "")
            if len(digit_features) != _UID_DIGIT_COUNT or not widget_ocr_uid:
                continue
            predicted_uid, confidence_floor = _predict_uid(model, digit_features)
            predicted_by_id[record_id] = (predicted_uid, confidence_floor)
            if predicted_uid == widget_ocr_uid:
                trusted_segmentable[record_id] = widget_ocr_uid
                new_ids.append(record_id)
        bootstrap_iterations.append(
            {
                "trustedSegmentableCount": len(trusted_segmentable),
                "newTrustedIds": new_ids,
            }
        )
        if not new_ids:
            break

    safe_resolved: Dict[str, str] = dict(safe_current_only)
    safe_resolved.update(trusted_segmentable)

    trusted_keep = 0
    trusted_fix = 0
    demoted = 0
    unresolved_by_reason: Counter[str] = Counter()

    for record in records:
        record_id = str(record.get("id") or "")
        current_uid = _current_uid(record)
        widget_ocr_uid = widget_ocr_index.get(record_id, "")
        digit_features = features_by_id.get(record_id) or []
        predicted_uid, confidence_floor = predicted_by_id.get(record_id, ("", 0.0))
        if not predicted_uid and len(digit_features) == _UID_DIGIT_COUNT and final_model is not None:
            predicted_uid, confidence_floor = _predict_uid(final_model, digit_features)

        resolved_uid = safe_resolved.get(record_id, "")
        status = "unresolved"
        reason = ""
        if resolved_uid:
            if resolved_uid == current_uid:
                trusted_keep += 1
                status = "trusted_keep"
                reason = "current_uid_matches_safe_resolution"
            else:
                trusted_fix += 1
                status = "trusted_fix"
                reason = "widget_ocr_matches_bootstrap_prediction"
            labels = record.get("labels")
            labels = labels if isinstance(labels, dict) else {}
            labels["uid_full"] = resolved_uid
            labels["uid"] = {"value": resolved_uid}
            labels["label"] = resolved_uid
            labels["reviewFinal"] = {
                "reviewer": str(args.reviewer_id),
                "reviewedAt": utc_now(),
                "notes": f"Recovered from UID widget OCR plus bootstrap digit classifier; widgetOcrUid={resolved_uid}.",
            }
            record["labels"] = labels
            record["qaStatus"] = "reviewed"
        else:
            if not widget_ocr_uid:
                reason = "missing_widget_ocr"
            elif len(digit_features) != _UID_DIGIT_COUNT:
                reason = "digit_segmentation_failed"
            elif predicted_uid and predicted_uid != widget_ocr_uid:
                reason = "ocr_model_disagree"
            else:
                reason = "unclassified"
            unresolved_by_reason[reason] += 1
            if not bool(args.keep_unresolved_reviewed):
                _clear_review(record)
                demoted += 1
        rows_by_id[record_id] = _record_summary(
            record,
            current_uid=current_uid,
            widget_ocr_uid=widget_ocr_uid,
            predicted_uid=predicted_uid,
            min_confidence=confidence_floor,
            status=status,
            resolved_uid=resolved_uid,
            reason=reason,
        )

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    ordered_rows = sorted(rows_by_id.values(), key=lambda item: (item["status"], item["sessionId"], item["path"]))
    output_json_path = Path(args.output_json).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": utc_now(),
        "manifest": str(manifest_path),
        "widgetOcrJson": str(Path(args.widget_ocr_json).resolve()),
        "workflow": str(args.workflow),
        "reviewedUidPanelCount": len(records),
        "seedTrustedCount": len(safe_current_only),
        "seedSegmentableTrustedCount": len([record_id for record_id in safe_current_only if len(features_by_id.get(record_id) or []) == _UID_DIGIT_COUNT]),
        "trustedSegmentableCount": len(trusted_segmentable),
        "trustedKeepCount": trusted_keep,
        "trustedFixCount": trusted_fix,
        "demotedUnresolvedCount": demoted,
        "unresolvedReasonCounts": dict(unresolved_by_reason),
        "bootstrapIterations": bootstrap_iterations,
        "records": ordered_rows,
        "dryRun": bool(args.dry_run),
    }
    output_json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    output_csv_path = Path(args.output_csv).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "sessionId",
                "path",
                "currentUidFull",
                "widgetOcrUidFull",
                "predictedUidFull",
                "predictionMinConfidence",
                "status",
                "resolvedUidFull",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(ordered_rows)

    print(
        json.dumps(
            {
                "reviewedUidPanelCount": len(records),
                "trustedKeepCount": trusted_keep,
                "trustedFixCount": trusted_fix,
                "demotedUnresolvedCount": demoted,
                "unresolvedReasonCounts": dict(unresolved_by_reason),
                "outputJson": str(output_json_path),
                "outputCsv": str(output_csv_path),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
