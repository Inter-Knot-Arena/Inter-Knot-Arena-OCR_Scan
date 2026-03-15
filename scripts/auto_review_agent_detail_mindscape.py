from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, AGENT_DETAIL_ROLE, filter_records, record_screen_role, source_index_from_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]

_MINDSCAPE_REGION = (0.58, 1.0, 0.0, 0.28)
_BRIGHT_THRESHOLD = 140
_BRIGHT_KERNEL = (11, 11)
_NUMERIC_THRESHOLDS = (50, 55, 60)
_GLYPH_SIZE = (20, 30)

_SAFE_RULES: dict[str, dict[str, float]] = {
    "0": {
        "numerator_score_max": 0.08,
        "numerator_margin_min": 0.03,
        "denominator_score_max": 0.07,
        "denominator_margin_min": 0.02,
    },
    "5": {
        "numerator_score_max": 0.03,
        "numerator_margin_min": 0.02,
        "denominator_score_max": 0.05,
        "denominator_margin_min": 0.02,
    },
}

_REFERENCE_BADGES: list[dict[str, Any]] = [
    {
        "path": Path(
            r"D:\IKA_DATA\ocr\raw\manual_screens\onedrive_account_20260308\agent_detail\onedrive_account_20260308_agent_detail_30c32a48ceae.png"
        ),
        "text": "5/6",
    },
    {
        "path": Path(r"D:\IKA_DATA\ocr\raw\manual_screens\batch_001_ru\agent_detail\batch_001_ru_agent_detail_00debaee2bb2.png"),
        "text": "0/6",
    },
    {
        "path": Path(r"D:\IKA_DATA\ocr\raw\manual_screens\batch_en_001\agent_detail\batch_en_001_agent_detail_d45c7c594a3e.png"),
        "text": "0/6",
    },
]


def _load_image_gray(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


def _bottom_left_region(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    y0 = int(round(height * _MINDSCAPE_REGION[0]))
    y1 = int(round(height * _MINDSCAPE_REGION[1]))
    x0 = int(round(width * _MINDSCAPE_REGION[2]))
    x1 = int(round(width * _MINDSCAPE_REGION[3]))
    region = image[y0:y1, x0:x1]
    return region if region.size else None


def _detect_label_box(region: np.ndarray) -> tuple[int, int, int, int] | None:
    bright = (region > _BRIGHT_THRESHOLD).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, _BRIGHT_KERNEL),
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)

    best_box: tuple[int, int, int, int] | None = None
    best_score: float | None = None
    width = region.shape[1]
    height = region.shape[0]
    for index in range(1, int(component_count)):
        x, y, w, h, area = [int(value) for value in stats[index]]
        if area <= 300:
            continue
        if x >= int(width * 0.5):
            continue
        score = area - abs(x - width * 0.12) * 40.0 - abs(y - height * 0.55) * 10.0
        if best_box is None or score > float(best_score):
            best_box = (x, y, w, h)
            best_score = score
    return best_box


def _extract_numeric_crop(region: np.ndarray, label_box: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, w, h = label_box
    x0 = max(0, int(round(x + w * 0.05)))
    x1 = min(region.shape[1], int(round(x + w * 0.78)))
    y0 = max(0, int(round(y + h * 0.55)))
    y1 = min(region.shape[0], int(round(y + h * 1.95)))
    crop = region[y0:y1, x0:x1]
    return crop if crop.size else None


def _extract_numeric_glyphs(crop: np.ndarray) -> tuple[list[np.ndarray], int] | None:
    for threshold in _NUMERIC_THRESHOLDS:
        binary = (crop < threshold).astype(np.uint8) * 255
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        glyphs: list[tuple[int, np.ndarray]] = []
        for index in range(1, int(component_count)):
            x, y, w, h, area = [int(value) for value in stats[index]]
            if area < 40 or h < 18 or w < 8:
                continue
            if x < int(crop.shape[1] * 0.30):
                continue
            if y < int(crop.shape[0] * 0.10) or y > int(crop.shape[0] * 0.75):
                continue
            glyphs.append((x, binary[y : y + h, x : x + w]))
        glyphs.sort(key=lambda item: item[0])
        if len(glyphs) == 3:
            return [item[1] for item in glyphs], threshold
    return None


def _normalize_glyph(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min((_GLYPH_SIZE[0] - 2) / max(width, 1), (_GLYPH_SIZE[1] - 2) / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_GLYPH_SIZE[1], _GLYPH_SIZE[0]), dtype=np.uint8)
    offset_x = (_GLYPH_SIZE[0] - resized_width) // 2
    offset_y = (_GLYPH_SIZE[1] - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return (canvas > 127).astype(np.float32)


def _build_digit_templates() -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {"0": [], "5": [], "6": []}
    for reference in _REFERENCE_BADGES:
        path = reference.get("path")
        text = str(reference.get("text") or "").strip()
        if not isinstance(path, Path) or len(text) != 3:
            continue
        image = _load_image_gray(path)
        if image is None:
            continue
        region = _bottom_left_region(image)
        if region is None:
            continue
        label_box = _detect_label_box(region)
        if label_box is None:
            continue
        crop = _extract_numeric_crop(region, label_box)
        if crop is None:
            continue
        extracted = _extract_numeric_glyphs(crop)
        if extracted is None:
            continue
        glyphs, _ = extracted
        if len(glyphs) != len(text):
            continue
        for glyph_image, character in zip(glyphs, text):
            if character not in templates:
                continue
            templates[character].append(_normalize_glyph(glyph_image))
    return templates


def _classify_digit(image: np.ndarray, templates: Mapping[str, Sequence[np.ndarray]]) -> tuple[str, float, float]:
    sample = _normalize_glyph(image)
    scores: list[tuple[float, str]] = []
    for label, variants in templates.items():
        if not variants:
            continue
        best_score = min(float(np.mean((sample - variant) ** 2)) for variant in variants)
        scores.append((best_score, label))
    if not scores:
        return "", float("inf"), 0.0
    scores.sort(key=lambda item: item[0])
    best_score, best_label = scores[0]
    margin = float(scores[1][0] - best_score) if len(scores) > 1 else 1.0
    return best_label, float(best_score), margin


def _predict_mindscape(image: np.ndarray, templates: Mapping[str, Sequence[np.ndarray]]) -> dict[str, Any] | None:
    region = _bottom_left_region(image)
    if region is None:
        return None
    label_box = _detect_label_box(region)
    if label_box is None:
        return None
    crop = _extract_numeric_crop(region, label_box)
    if crop is None:
        return None
    extracted = _extract_numeric_glyphs(crop)
    if extracted is None:
        return None
    glyphs, threshold = extracted
    numerator_label, numerator_score, numerator_margin = _classify_digit(glyphs[0], templates)
    denominator_label, denominator_score, denominator_margin = _classify_digit(glyphs[2], templates)
    return {
        "numerator": numerator_label,
        "numeratorScore": numerator_score,
        "numeratorMargin": numerator_margin,
        "denominator": denominator_label,
        "denominatorScore": denominator_score,
        "denominatorMargin": denominator_margin,
        "threshold": threshold,
        "labelBox": {
            "x": label_box[0],
            "y": label_box[1],
            "width": label_box[2],
            "height": label_box[3],
        },
    }


def _is_safe_prediction(prediction: Mapping[str, Any]) -> bool:
    numerator = str(prediction.get("numerator") or "")
    denominator = str(prediction.get("denominator") or "")
    if denominator != "6":
        return False
    rule = _SAFE_RULES.get(numerator)
    if not rule:
        return False
    return (
        float(prediction.get("numeratorScore") or 1.0) <= rule["numerator_score_max"]
        and float(prediction.get("numeratorMargin") or 0.0) >= rule["numerator_margin_min"]
        and float(prediction.get("denominatorScore") or 1.0) <= rule["denominator_score_max"]
        and float(prediction.get("denominatorMargin") or 0.0) >= rule["denominator_margin_min"]
    )


def _reviewed_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return {}
    if str(record.get("qaStatus") or "").strip().lower() != "reviewed":
        return {}
    if not isinstance(labels.get("reviewFinal"), dict):
        return {}
    return labels


def _update_snapshot(labels: Dict[str, Any], mindscape: int, cap: int) -> None:
    snapshot = labels.get("agentSnapshot")
    snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
    agent_id = str(labels.get("agent_icon_id") or "").strip()
    if agent_id:
        snapshot["agentId"] = agent_id
    if labels.get("agent_level") is not None:
        snapshot["level"] = labels.get("agent_level")
    if labels.get("agent_level_cap") is not None:
        snapshot["levelCap"] = labels.get("agent_level_cap")
    if isinstance(labels.get("agent_stats"), dict):
        snapshot["stats"] = labels.get("agent_stats")
    snapshot["mindscape"] = mindscape
    snapshot["mindscapeCap"] = cap
    labels["agentSnapshot"] = snapshot


def _update_review_final(labels: Dict[str, Any], reviewer_id: str) -> None:
    review_final = labels.get("reviewFinal")
    review_final = dict(review_final) if isinstance(review_final, dict) else {}
    note = "Auto-reviewed agent mindscape from lower-left CINEMA footer; adjacent STR counter ignored."
    prior_notes = str(review_final.get("notes") or "").strip()
    if note not in prior_notes:
        review_final["notes"] = f"{prior_notes} {note}".strip()
    review_final["reviewer"] = reviewer_id
    review_final["reviewedAt"] = utc_now()
    labels["reviewFinal"] = review_final


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-review agent_detail mindscape values from lower-left CINEMA footer.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--reviewer-id", default="auto_review_agent_detail_mindscape")
    parser.add_argument("--output-json", default="docs/auto_review_agent_detail_mindscape.json")
    parser.add_argument("--dry-run", action="store_true")
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
        import_eligible_only=True,
    )

    templates = _build_digit_templates()
    if not all(templates.get(label) for label in ("0", "5", "6")):
        coverage = {key: len(value) for key, value in templates.items()}
        raise RuntimeError(f"Insufficient mindscape template coverage: {coverage}")

    processed = 0
    applied = 0
    skipped_existing = 0
    skipped_missing_assets = 0
    skipped_nonreviewed = 0
    skipped_derived = 0
    skipped_parse_failed = 0
    skipped_unsafe = 0
    prediction_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        if record_screen_role(record, source_index) != AGENT_DETAIL_ROLE:
            continue
        if str(record.get("kind") or "").strip().lower() != "manual_screenshot":
            skipped_derived += 1
            continue

        labels = _reviewed_labels(record)
        if not labels:
            skipped_nonreviewed += 1
            continue

        if labels.get("agent_mindscape") is not None and labels.get("agent_mindscape_cap") is not None:
            skipped_existing += 1
            continue

        path_value = str(record.get("path") or "").strip()
        if not path_value:
            skipped_missing_assets += 1
            continue

        image = _load_image_gray(Path(path_value))
        if image is None:
            skipped_missing_assets += 1
            continue

        processed += 1
        prediction = _predict_mindscape(image, templates)
        if prediction is None:
            skipped_parse_failed += 1
            continue

        prediction_key = f"{prediction['numerator']}/{prediction['denominator']}"
        prediction_counts[prediction_key] += 1
        if not _is_safe_prediction(prediction):
            skipped_unsafe += 1
            continue

        mindscape_value = int(str(prediction["numerator"]))
        labels["agent_mindscape"] = mindscape_value
        labels["agent_mindscape_cap"] = 6
        _update_snapshot(labels, mindscape_value, 6)
        _update_review_final(labels, args.reviewer_id)
        applied += 1
        applied_counts[f"{mindscape_value}/6"] += 1

        if len(examples) < 12:
            examples.append(
                {
                    "recordId": str(record.get("id") or ""),
                    "focusAgentId": str(record.get("focusAgentId") or ""),
                    "mindscape": mindscape_value,
                    "mindscapeCap": 6,
                    "numeratorScore": round(float(prediction["numeratorScore"]), 4),
                    "numeratorMargin": round(float(prediction["numeratorMargin"]), 4),
                    "denominatorScore": round(float(prediction["denominatorScore"]), 4),
                    "denominatorMargin": round(float(prediction["denominatorMargin"]), 4),
                    "path": path_value,
                }
            )

    report = {
        "generatedAt": utc_now(),
        "workflow": args.workflow,
        "reviewerId": args.reviewer_id,
        "processedRecords": processed,
        "appliedRecords": applied,
        "skippedExisting": skipped_existing,
        "skippedMissingAssets": skipped_missing_assets,
        "skippedNonReviewed": skipped_nonreviewed,
        "skippedDerived": skipped_derived,
        "skippedParseFailed": skipped_parse_failed,
        "skippedUnsafe": skipped_unsafe,
        "digitTemplates": {key: len(value) for key, value in sorted(templates.items())},
        "predictionCounts": dict(sorted(prediction_counts.items())),
        "appliedCounts": dict(sorted(applied_counts.items())),
        "safeRules": _SAFE_RULES,
        "examples": examples,
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
