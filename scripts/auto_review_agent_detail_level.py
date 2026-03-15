from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, AGENT_DETAIL_ROLE, filter_records, record_screen_role, source_index_from_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]

_PANEL_BOX = (0.52, 0.32, 0.96, 0.78)
_STAT_FIELDS: list[tuple[str, str]] = [
    ("hp", "atk"),
    ("def", "impact"),
    ("crit_rate", "crit_dmg"),
    ("anomaly_mastery", "anomaly_prof"),
    ("pen_ratio", "energy_regen"),
]
_LEVEL_CURRENT_SLICE = (0.18, 0.50)
_LEVEL_CAP_SLICE = (0.44, 0.78)
_CURRENT_THRESHOLDS = (170, 180, 190, 200)
_CAP_THRESHOLDS = (15, 20, 25)
_GLYPH_SIZE = (20, 30)
_VALID_LEVEL_CAPS = {10, 20, 30, 40, 50, 60}

_REFERENCE_STATS: list[dict[str, Any]] = [
    {
        "path": Path(r"D:\IKA_DATA\ocr\raw\manual_screens\onedrive_account_20260308\agent_detail\onedrive_account_20260308_agent_detail_30c32a48ceae.png"),
        "fields": {
            "hp": "12051",
            "atk": "2747",
            "crit_rate": "82,6%",
            "anomaly_mastery": "94",
        },
    },
    {
        "path": Path(r"D:\IKA_DATA\ocr\raw\manual_screens\batch_001_ru\agent_detail\batch_001_ru_agent_detail_00debaee2bb2.png"),
        "fields": {
            "atk": "3465",
        },
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


def _threshold_binary(image: np.ndarray, threshold: int, *, invert: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, output = cv2.threshold(image, threshold, 255, mode)
    return output


def _panel_origin(image: np.ndarray) -> tuple[int, int, np.ndarray]:
    height, width = image.shape[:2]
    x0 = int(round(width * _PANEL_BOX[0]))
    y0 = int(round(height * _PANEL_BOX[1]))
    x1 = int(round(width * _PANEL_BOX[2]))
    y1 = int(round(height * _PANEL_BOX[3]))
    return x0, y0, image[y0:y1, x0:x1]


def _detect_level_boxes(image: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    origin_x, origin_y, panel = _panel_origin(image)
    dark_mask = (panel < 35).astype(np.uint8) * 255
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    panel_width = panel.shape[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 20_000 or w < 300 or h < 60:
            continue
        if w > int(panel_width * 0.8):
            continue
        boxes.append((origin_x + x, origin_y + y, w, h))

    boxes.sort(key=lambda item: (item[1], item[0]))
    if len(boxes) < 2:
        return None

    first, second = boxes[0], boxes[1]
    if first[0] < second[0]:
        return first, second
    return second, first


def _connected_components(
    binary: np.ndarray,
    *,
    area_min: int,
    height_min: int,
    width_min: int,
) -> list[tuple[int, int, int, int]]:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    output: list[tuple[int, int, int, int]] = []
    for index in range(1, int(component_count)):
        x, y, w, h, area = [int(value) for value in stats[index]]
        if area < area_min or h < height_min or w < width_min:
            continue
        output.append((x, y, w, h))
    output.sort(key=lambda item: item[0])
    return output


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


def _merge_overlapping_boxes(boxes: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        if not merged:
            merged.append((x, y, x + w, y + h))
            continue
        prev_x0, prev_y0, prev_x1, prev_y1 = merged[-1]
        overlap = min(prev_x1, x + w) - max(prev_x0, x)
        if overlap >= 0:
            merged[-1] = (min(prev_x0, x), min(prev_y0, y), max(prev_x1, x + w), max(prev_y1, y + h))
            continue
        merged.append((x, y, x + w, y + h))
    return merged


def _glyph_images_from_block(binary: np.ndarray) -> list[np.ndarray]:
    components = _connected_components(binary, area_min=10, height_min=6, width_min=1)
    glyph_boxes = _merge_overlapping_boxes(components)
    images: list[np.ndarray] = []
    for x0, y0, x1, y1 in glyph_boxes:
        images.append(binary[y0:y1, x0:x1])
    return images


def _classify_digit(image: np.ndarray, templates: Mapping[str, list[np.ndarray]]) -> tuple[str, float]:
    sample = _normalize_glyph(image)
    best_label = ""
    best_score = float("inf")
    for label, variants in templates.items():
        for variant in variants:
            score = float(np.mean((sample - variant) ** 2))
            if score < best_score:
                best_label = label
                best_score = score
    return best_label, best_score


def _extract_current_digit_components(level_crop: np.ndarray) -> list[np.ndarray]:
    width = level_crop.shape[1]
    x0 = int(round(width * _LEVEL_CURRENT_SLICE[0]))
    x1 = int(round(width * _LEVEL_CURRENT_SLICE[1]))
    current_crop = level_crop[:, x0:x1]

    for threshold in _CURRENT_THRESHOLDS:
        binary = _threshold_binary(current_crop, threshold, invert=False)
        components = _connected_components(binary, area_min=10, height_min=6, width_min=1)
        components = [component for component in components if component[0] >= int(binary.shape[1] * 0.25)]
        if len(components) == 2:
            return [binary[y : y + h, x : x + w] for x, y, w, h in components]
    return []


def _extract_cap_digit_components(level_crop: np.ndarray) -> list[np.ndarray]:
    width = level_crop.shape[1]
    x0 = int(round(width * _LEVEL_CAP_SLICE[0]))
    x1 = int(round(width * _LEVEL_CAP_SLICE[1]))
    cap_crop = level_crop[:, x0:x1]
    crop_area = cap_crop.shape[0] * cap_crop.shape[1]

    for threshold in _CAP_THRESHOLDS:
        binary = (cap_crop < threshold).astype(np.uint8) * 255
        components = _connected_components(binary, area_min=20, height_min=8, width_min=2)
        filtered: list[tuple[int, int, int, int]] = []
        for x, y, w, h in components:
            touches_edge = x <= 1 or y <= 1 or (x + w) >= (binary.shape[1] - 1) or (y + h) >= (binary.shape[0] - 1)
            if touches_edge and (w * h) > int(crop_area * 0.15):
                continue
            filtered.append((x, y, w, h))
        if len(filtered) == 2:
            return [binary[y : y + h, x : x + w] for x, y, w, h in filtered]
    return []


def _extract_reference_value_block(image: np.ndarray, field_name: str) -> np.ndarray | None:
    boxes = _detect_level_boxes(image)
    if boxes is None:
        return None
    left_box, right_box = boxes
    average_height = int(round((left_box[3] + right_box[3]) / 2.0))
    row_top = left_box[1] + left_box[3] + int(round(average_height * 0.22))
    row_height = max(38, int(round(average_height * 0.57)))
    row_gap = max(6, int(round(average_height * 0.12)))

    field_index = -1
    use_left = True
    for index, row_fields in enumerate(_STAT_FIELDS):
        if field_name == row_fields[0]:
            field_index = index
            use_left = True
            break
        if field_name == row_fields[1]:
            field_index = index
            use_left = False
            break
    if field_index < 0:
        return None

    row_y = row_top + field_index * (row_height + row_gap)
    box = left_box if use_left else right_box
    row = image[row_y : row_y + row_height, box[0] : box[0] + box[2]]
    if row.size == 0:
        return None

    binary = _threshold_binary(row, 150, invert=False)
    components = _connected_components(binary, area_min=10, height_min=6, width_min=1)
    if not components:
        return None

    x0, y0, w0, h0 = components[-1]
    left = x0
    top = y0
    right = x0 + w0
    bottom = y0 + h0
    for x, y, w, h in reversed(components[:-1]):
        gap = left - (x + w)
        overlap = min(bottom, y + h) - max(top, y)
        if gap <= max(w, right - left) * 0.12 + 16 and overlap > min(h, bottom - top) * 0.35:
            left = min(left, x)
            top = min(top, y)
            right = max(right, x + w)
            bottom = max(bottom, y + h)
            continue
        break

    pad = 6
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(binary.shape[1], right + pad)
    bottom = min(binary.shape[0], bottom + pad)
    return binary[top:bottom, left:right]


def _build_digit_templates() -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {}

    for reference in _REFERENCE_STATS:
        image = _load_image_gray(reference["path"])
        if image is None:
            continue
        fields = reference.get("fields")
        if not isinstance(fields, dict):
            continue
        for field_name, raw_value in fields.items():
            if not isinstance(field_name, str) or not isinstance(raw_value, str):
                continue
            block = _extract_reference_value_block(image, field_name)
            if block is None:
                continue
            glyph_images = _glyph_images_from_block(block)
            if len(glyph_images) != len(raw_value):
                continue
            for glyph_image, character in zip(glyph_images, raw_value):
                if not character.isdigit():
                    continue
                templates.setdefault(character, []).append(_normalize_glyph(glyph_image))

    for reference in _REFERENCE_STATS:
        image = _load_image_gray(reference["path"])
        if image is None:
            continue
        boxes = _detect_level_boxes(image)
        if boxes is None:
            continue
        left_box, _ = boxes
        x, y, w, h = left_box
        level_crop = image[y : y + h, x : x + w]
        current_digits = _extract_current_digit_components(level_crop)
        cap_digits = _extract_cap_digit_components(level_crop)
        if len(current_digits) == 2:
            for glyph_image, character in zip(current_digits, "60"):
                templates.setdefault(character, []).append(_normalize_glyph(glyph_image))
        if len(cap_digits) == 2:
            for glyph_image, character in zip(cap_digits, "60"):
                templates.setdefault(character, []).append(_normalize_glyph(glyph_image))

    return templates


def _parse_level_pair(image: np.ndarray, templates: Mapping[str, list[np.ndarray]]) -> tuple[int, int] | None:
    boxes = _detect_level_boxes(image)
    if boxes is None:
        return None
    left_box, _ = boxes
    x, y, w, h = left_box
    level_crop = image[y : y + h, x : x + w]
    if level_crop.size == 0:
        return None

    current_digits = _extract_current_digit_components(level_crop)
    cap_digits = _extract_cap_digit_components(level_crop)
    if len(current_digits) != 2 or len(cap_digits) != 2:
        return None

    current_text = "".join(_classify_digit(item, templates)[0] for item in current_digits)
    cap_text = "".join(_classify_digit(item, templates)[0] for item in cap_digits)
    if not (current_text.isdigit() and cap_text.isdigit()):
        return None

    current = int(current_text)
    cap = int(cap_text)
    if current <= 0 or cap <= 0 or current > cap or current > 60:
        return None
    if cap not in _VALID_LEVEL_CAPS:
        return None
    return current, cap


def _reviewed_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return {}
    if str(record.get("qaStatus") or "").strip().lower() != "reviewed":
        return {}
    if not isinstance(labels.get("reviewFinal"), dict):
        return {}
    return labels


def _update_review_final(labels: Dict[str, Any], reviewer_id: str) -> None:
    review_final = labels.get("reviewFinal")
    review_final = dict(review_final) if isinstance(review_final, dict) else {}
    note = "Auto-reviewed agent level pair from detected level pill."
    prior_notes = str(review_final.get("notes") or "").strip()
    if note not in prior_notes:
        review_final["notes"] = f"{prior_notes} {note}".strip()
    review_final["reviewer"] = reviewer_id
    review_final["reviewedAt"] = utc_now()
    labels["reviewFinal"] = review_final


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-review agent_detail level pairs from detected level pill snapshots.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--reviewer-id", default="auto_review_agent_detail_level")
    parser.add_argument("--output-json", default="docs/auto_review_agent_detail_level.json")
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
    if len(templates) < 10:
        raise RuntimeError(f"Insufficient digit template coverage: built {sorted(templates.keys())}")

    processed = 0
    applied = 0
    skipped_existing = 0
    skipped_missing_assets = 0
    skipped_nonreviewed = 0
    skipped_parse_failed = 0
    examples: list[dict[str, Any]] = []

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        if record_screen_role(record, source_index) != AGENT_DETAIL_ROLE:
            continue

        labels = _reviewed_labels(record)
        if not labels:
            skipped_nonreviewed += 1
            continue

        if labels.get("agent_level") is not None and labels.get("agent_level_cap") is not None:
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
        level_pair = _parse_level_pair(image, templates)
        if level_pair is None:
            skipped_parse_failed += 1
            continue

        current, cap = level_pair
        labels["agent_level"] = current
        labels["agent_level_cap"] = cap
        _update_review_final(labels, args.reviewer_id)
        applied += 1

        if len(examples) < 12:
            examples.append(
                {
                    "recordId": str(record.get("id") or ""),
                    "agentLevel": current,
                    "agentLevelCap": cap,
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
        "skippedParseFailed": skipped_parse_failed,
        "digitTemplates": {key: len(value) for key, value in sorted(templates.items())},
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
