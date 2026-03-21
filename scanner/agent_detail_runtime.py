from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np

from .model_runtime import ModelRegistry, classify_uid_digit

_PANEL_BOX = (0.52, 0.32, 0.96, 0.78)
_MINDSCAPE_REGION = (0.58, 1.0, 0.0, 0.28)
_LEVEL_CURRENT_SLICE = (0.18, 0.50)
_LEVEL_CAP_SLICE = (0.44, 0.70)
_CURRENT_THRESHOLDS = (170, 180, 190, 200)
_CAP_THRESHOLDS = (15, 20, 25)
_NUMERIC_THRESHOLDS = (50, 55, 60)
_VALID_LEVEL_CAPS = {10, 20, 30, 40, 50, 60}
_LEVEL_DIGIT_COUNT_BONUS = 0.24
_DIGIT_TEMPLATE_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contracts" / "agent-detail-digit-templates.json"
_TEMPLATE_GLYPH_SIZE = (20, 30)
_STAT_FIELDS: list[tuple[str, str]] = [
    ("hp", "atk"),
    ("def", "impact"),
    ("crit_rate", "crit_dmg"),
    ("anomaly_mastery", "anomaly_prof"),
    ("pen_ratio", "energy_regen"),
]
_STAT_SPECS: list[dict[str, Any]] = [
    {"source": "hp", "canonical": "hp_flat", "kind": "int", "min": 1000, "max": 30000},
    {"source": "atk", "canonical": "attack_flat", "kind": "int", "min": 100, "max": 6000},
    {"source": "def", "canonical": "defense_flat", "kind": "int", "min": 100, "max": 4000},
    {"source": "impact", "canonical": "impact", "kind": "int", "min": 1, "max": 400},
    {"source": "crit_rate", "canonical": "crit_rate_pct", "kind": "percent", "min": 0, "max": 100},
    {"source": "crit_dmg", "canonical": "crit_damage_pct", "kind": "percent", "min": 0, "max": 400},
    {"source": "anomaly_mastery", "canonical": "anomaly_mastery", "kind": "int", "min": 0, "max": 600},
    {"source": "anomaly_prof", "canonical": "anomaly_proficiency", "kind": "int", "min": 0, "max": 600},
    {"source": "pen_ratio", "canonical": "pen_ratio_pct", "kind": "percent", "min": 0, "max": 100},
    {"source": "energy_regen", "canonical": "energy_regen", "kind": "decimal", "min": 0, "max": 100},
]


@dataclass(slots=True)
class AgentDetailReading:
    level: int | None
    level_cap: int | None
    level_confidence: float
    mindscape: int | None
    mindscape_cap: int | None
    mindscape_confidence: float
    stats: Dict[str, int | float]
    stats_confidence: float
    low_conf_reasons: List[str]


@lru_cache(maxsize=1)
def _digit_template_index() -> Dict[str, list[np.ndarray]]:
    if not _DIGIT_TEMPLATE_CONTRACT_PATH.exists():
        return {}
    payload = json.loads(_DIGIT_TEMPLATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    raw_templates = payload.get("templates")
    if not isinstance(raw_templates, dict):
        return {}

    index: Dict[str, list[np.ndarray]] = {}
    for label, variants in raw_templates.items():
        if not isinstance(variants, list):
            continue
        parsed_variants: list[np.ndarray] = []
        for variant in variants:
            if not isinstance(variant, list):
                continue
            rows = [str(row) for row in variant if isinstance(row, str)]
            if not rows:
                continue
            matrix = np.array([[1.0 if char == "1" else 0.0 for char in row] for row in rows], dtype=np.float32)
            parsed_variants.append(matrix)
        if parsed_variants:
            index[str(label)] = parsed_variants
    return index


def _normalize_template_glyph(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min((_TEMPLATE_GLYPH_SIZE[0] - 2) / max(width, 1), (_TEMPLATE_GLYPH_SIZE[1] - 2) / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_TEMPLATE_GLYPH_SIZE[1], _TEMPLATE_GLYPH_SIZE[0]), dtype=np.uint8)
    offset_x = (_TEMPLATE_GLYPH_SIZE[0] - resized_width) // 2
    offset_y = (_TEMPLATE_GLYPH_SIZE[1] - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return (canvas > 127).astype(np.float32)


def _template_confidence(score: float, margin: float) -> float:
    score_factor = max(0.0, 1.0 - (min(float(score), 0.25) / 0.25))
    margin_factor = min(max(float(margin), 0.0), 0.12) / 0.12
    return min(0.999, 0.72 + (0.18 * score_factor) + (0.10 * margin_factor))


def _classify_template_digit(image: np.ndarray) -> tuple[str, float, float]:
    templates = _digit_template_index()
    if not templates:
        return "", float("inf"), 0.0
    sample = _normalize_template_glyph(image)
    scores: list[tuple[float, str]] = []
    for label, variants in templates.items():
        if not variants:
            continue
        best_score = min(float(((sample - variant) ** 2).mean()) for variant in variants)
        scores.append((best_score, label))
    if not scores:
        return "", float("inf"), 0.0
    scores.sort(key=lambda item: item[0])
    best_score, best_label = scores[0]
    margin = float(scores[1][0] - best_score) if len(scores) > 1 else 1.0
    return best_label, best_score, margin


def _threshold_binary(image: np.ndarray, threshold: int, *, invert: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, output = cv2.threshold(image, threshold, 255, mode)
    return output


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
        x, y, width, height, area = [int(value) for value in stats[index]]
        if area < area_min or height < height_min or width < width_min:
            continue
        output.append((x, y, width, height))
    output.sort(key=lambda item: item[0])
    return output


def _merge_overlapping_boxes(boxes: Sequence[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for x, y, width, height in boxes:
        x1 = x + width
        y1 = y + height
        if not merged:
            merged.append((x, y, x1, y1))
            continue
        prev_x0, prev_y0, prev_x1, prev_y1 = merged[-1]
        overlap = min(prev_x1, x1) - max(prev_x0, x)
        if overlap >= 0:
            merged[-1] = (min(prev_x0, x), min(prev_y0, y), max(prev_x1, x1), max(prev_y1, y1))
            continue
        merged.append((x, y, x1, y1))
    return merged


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
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < 20_000 or width < 300 or height < 60:
            continue
        if width > int(panel_width * 0.8):
            continue
        boxes.append((origin_x + x, origin_y + y, width, height))

    boxes.sort(key=lambda item: (item[1], item[0]))
    if len(boxes) < 2:
        return None
    first, second = boxes[0], boxes[1]
    return (first, second) if first[0] < second[0] else (second, first)


def _classify_digit_component(image: np.ndarray) -> tuple[str, float]:
    if not ModelRegistry.has_uid_model():
        return "", 0.0
    prediction = classify_uid_digit(image)
    return prediction.label, float(prediction.confidence)


def _classify_components(
    binary: np.ndarray,
    components: Sequence[tuple[int, int, int, int]],
) -> tuple[str, float]:
    labels: list[str] = []
    confidences: list[float] = []
    for x, y, width, height in components:
        glyph = binary[y : y + height, x : x + width]
        label, confidence = _classify_digit_component(glyph)
        if not label.isdigit():
            return "", 0.0
        labels.append(label)
        confidences.append(confidence)
    if not labels:
        return "", 0.0
    return "".join(labels), float(sum(confidences) / len(confidences))


def _classify_level_digit_component(image: np.ndarray) -> tuple[str, float]:
    label, confidence = _classify_digit_component(image)
    if label == "7" and confidence <= 0.75:
        height, width = image.shape[:2]
        fill_ratio = float(np.count_nonzero(image)) / float(image.size or 1)
        if (width <= 14 and height >= 28) or (
            width <= max(16, int(round(height * 0.5))) and fill_ratio <= 0.55
        ):
            return "1", max(float(confidence), 0.55)
    return label, float(confidence)


def _classify_level_components(
    binary: np.ndarray,
    components: Sequence[tuple[int, int, int, int]],
) -> tuple[str, float]:
    labels: list[str] = []
    confidences: list[float] = []
    for x, y, width, height in components:
        glyph = binary[y : y + height, x : x + width]
        label, confidence = _classify_level_digit_component(glyph)
        if not label.isdigit():
            return "", 0.0
        labels.append(label)
        confidences.append(confidence)
    if not labels:
        return "", 0.0
    return "".join(labels), float(sum(confidences) / len(confidences))


def _parse_level_part(
    crop: np.ndarray,
    *,
    thresholds: Sequence[int],
    invert: bool,
    min_digits: int,
    max_digits: int,
    min_component_x_fraction: float,
    select_from: str = "right",
) -> tuple[int | None, float]:
    best_value: int | None = None
    best_confidence = 0.0
    best_score = float("-inf")
    if select_from not in {"left", "right"}:
        raise ValueError(f"unsupported level component selection: {select_from}")
    for threshold in thresholds:
        binary = _threshold_binary(crop, threshold, invert=invert)
        components = _connected_components(binary, area_min=10, height_min=6, width_min=1)
        components = [
            component
            for component in components
            if component[0] >= int(binary.shape[1] * min_component_x_fraction)
        ]
        if not components:
            continue
        for digit_count in range(max_digits, min_digits - 1, -1):
            if len(components) < digit_count:
                continue
            candidate = components[:digit_count] if select_from == "left" else components[-digit_count:]
            text, confidence = _classify_level_components(binary, candidate)
            if not text.isdigit():
                continue
            value = int(text)
            score = float(confidence) + max(0, digit_count - min_digits) * _LEVEL_DIGIT_COUNT_BONUS
            if score > best_score or (score == best_score and confidence > best_confidence):
                best_value = value
                best_confidence = confidence
                best_score = score
    return best_value, best_confidence


def _parse_level_cap_from_tens_digit(crop: np.ndarray) -> tuple[int | None, float]:
    height, width = crop.shape[:2]
    if height <= 0 or width <= 0:
        return None, 0.0

    best_value: int | None = None
    best_confidence = 0.0
    for threshold in _CAP_THRESHOLDS:
        binary = _threshold_binary(crop, threshold, invert=True)
        components = _connected_components(binary, area_min=10, height_min=6, width_min=1)
        components = [
            component
            for component in components
            if component[1] >= int(height * 0.18)
            and component[2] <= int(width * 0.55)
            and component[3] >= int(height * 0.45)
        ]
        if not components:
            continue
        components.sort(key=lambda component: component[0])
        for x, y, box_width, box_height in components:
            glyph = binary[y : y + box_height, x : x + box_width]
            label, confidence = _classify_level_digit_component(glyph)
            if label not in {"1", "2", "3", "4", "5", "6"}:
                continue
            cap = int(label) * 10
            boosted_confidence = min(0.995, max(float(confidence), 0.86))
            if boosted_confidence > best_confidence:
                best_value = cap
                best_confidence = boosted_confidence
            break
    return best_value, best_confidence


def _extract_level(image: np.ndarray) -> tuple[int | None, int | None, float, list[str]]:
    reasons: list[str] = []
    boxes = _detect_level_boxes(image)
    if boxes is None:
        return None, None, 0.0, ["agent_detail_level_boxes_missing"]
    left_box, _ = boxes
    x, y, width, height = left_box
    level_crop = image[y : y + height, x : x + width]
    if level_crop.size == 0:
        return None, None, 0.0, ["agent_detail_level_crop_empty"]

    current_x0 = int(round(width * _LEVEL_CURRENT_SLICE[0]))
    current_x1 = int(round(width * _LEVEL_CURRENT_SLICE[1]))
    cap_x0 = int(round(width * _LEVEL_CAP_SLICE[0]))
    cap_x1 = int(round(width * _LEVEL_CAP_SLICE[1]))

    current_crop = level_crop[:, current_x0:current_x1]
    cap_crop = level_crop[:, cap_x0:cap_x1]

    current, current_confidence = _parse_level_part(
        current_crop,
        thresholds=_CURRENT_THRESHOLDS,
        invert=False,
        min_digits=1,
        max_digits=2,
        min_component_x_fraction=0.25,
        select_from="right",
    )
    cap, cap_confidence = _parse_level_part(
        cap_crop,
        thresholds=_CAP_THRESHOLDS,
        invert=True,
        min_digits=2,
        max_digits=2,
        min_component_x_fraction=0.0,
        select_from="right",
    )
    if cap is None or cap not in _VALID_LEVEL_CAPS:
        fallback_cap, fallback_cap_confidence = _parse_level_cap_from_tens_digit(cap_crop)
        if fallback_cap is not None:
            cap = fallback_cap
            cap_confidence = max(float(cap_confidence), float(fallback_cap_confidence))

    if current is None:
        reasons.append("agent_detail_level_current_missing")
    if cap is None:
        reasons.append("agent_detail_level_cap_missing")
    if cap is not None and cap not in _VALID_LEVEL_CAPS:
        reasons.append(f"agent_detail_level_cap_invalid:{cap}")
        cap = None
        cap_confidence = 0.0
    if current is not None and current <= 0:
        reasons.append(f"agent_detail_level_current_invalid:{current}")
        current = None
        current_confidence = 0.0
    if current is not None and cap is not None and current > cap:
        reasons.append(f"agent_detail_level_current_gt_cap:{current}/{cap}")
        current = None
        current_confidence = 0.0
    if current is not None and cap is None and current in _VALID_LEVEL_CAPS:
        cap = current
        cap_confidence = max(float(cap_confidence), min(float(current_confidence), 0.82))

    confidence_values = [value for value in (current_confidence, cap_confidence) if value > 0.0]
    confidence = float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
    return current, cap, confidence, reasons


def _bottom_left_region(image: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    y0 = int(round(height * _MINDSCAPE_REGION[0]))
    y1 = int(round(height * _MINDSCAPE_REGION[1]))
    x0 = int(round(width * _MINDSCAPE_REGION[2]))
    x1 = int(round(width * _MINDSCAPE_REGION[3]))
    region = image[y0:y1, x0:x1]
    return region if region.size else None


def _detect_mindscape_label_box(region: np.ndarray) -> tuple[int, int, int, int] | None:
    bright = (region > 140).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    best_box: tuple[int, int, int, int] | None = None
    best_score: float | None = None
    width = region.shape[1]
    height = region.shape[0]
    for index in range(1, int(component_count)):
        x, y, box_width, box_height, area = [int(value) for value in stats[index]]
        if area <= 300:
            continue
        if x >= int(width * 0.5):
            continue
        score = area - abs(x - width * 0.12) * 40.0 - abs(y - height * 0.55) * 10.0
        if best_box is None or score > float(best_score):
            best_box = (x, y, box_width, box_height)
            best_score = score
    return best_box


def _extract_mindscape_numeric_crop(region: np.ndarray, label_box: tuple[int, int, int, int]) -> np.ndarray | None:
    x, y, width, height = label_box
    x0 = max(0, int(round(x + width * 0.05)))
    x1 = min(region.shape[1], int(round(x + width * 0.78)))
    y0 = max(0, int(round(y + height * 0.55)))
    y1 = min(region.shape[0], int(round(y + height * 1.95)))
    crop = region[y0:y1, x0:x1]
    return crop if crop.size else None


def _extract_mindscape_numeric_crop_fixed(region: np.ndarray) -> np.ndarray | None:
    height, width = region.shape[:2]
    x0 = max(0, int(round(width * 0.14)))
    x1 = min(width, int(round(width * 0.34)))
    y0 = max(0, int(round(height * 0.72)))
    y1 = min(height, int(round(height * 0.91)))
    crop = region[y0:y1, x0:x1]
    return crop if crop.size else None


def _mindscape_fill_ratio(binary: np.ndarray, component: tuple[int, int, int, int]) -> float:
    x, y, width, height = component
    glyph = binary[y : y + height, x : x + width]
    return float(np.count_nonzero(glyph)) / float(glyph.size or 1)


def _looks_zero_mindscape_numerator(
    binary: np.ndarray,
    components: Sequence[tuple[int, int, int, int]],
    numerator_text: str,
) -> bool:
    if len(components) != 3 or numerator_text not in {"8", "9"}:
        return False

    numerator_box, separator_box, denominator_box = components
    numerator_fill = _mindscape_fill_ratio(binary, numerator_box)
    separator_fill = _mindscape_fill_ratio(binary, separator_box)
    denominator_fill = _mindscape_fill_ratio(binary, denominator_box)

    return (
        separator_box[0] > numerator_box[0]
        and denominator_box[0] > separator_box[0]
        and numerator_box[2] >= 24
        and numerator_box[3] >= 28
        and separator_box[2] <= max(24, int(round(numerator_box[2] * 0.9)))
        and separator_box[3] >= int(round(numerator_box[3] * 0.9))
        and 0.55 <= numerator_fill <= 0.74
        and separator_fill <= 0.52
        and denominator_fill >= 0.45
    )


def _parse_mindscape_numeric_crop(crop: np.ndarray) -> tuple[int | None, int | None, float, bool]:
    best_value: int | None = None
    best_confidence = 0.0
    denominator_ok = False
    for threshold in _NUMERIC_THRESHOLDS:
        binary = (crop < threshold).astype(np.uint8) * 255
        components = _connected_components(binary, area_min=40, height_min=18, width_min=8)
        components = [
            component
            for component in components
            if component[0] >= int(crop.shape[1] * 0.20)
            and component[1] >= int(crop.shape[0] * 0.35)
            and component[1] <= int(crop.shape[0] * 0.88)
            and component[3] >= 24
        ]
        if len(components) != 3:
            continue
        numerator_text, numerator_confidence = _classify_components(binary, [components[0]])
        denominator_text, denominator_confidence = _classify_components(binary, [components[2]])
        if not numerator_text.isdigit():
            continue
        numerator_value = int(numerator_text)
        if not 0 <= numerator_value <= 6:
            if _looks_zero_mindscape_numerator(binary, components, numerator_text):
                numerator_value = 0
                numerator_confidence = max(float(numerator_confidence), 0.68)
            else:
                continue
        denominator_value = int(denominator_text) if denominator_text.isdigit() else None
        structural_denominator_ok = (
            components[2][0] > components[1][0] > components[0][0]
            and components[2][2] >= 18
            and components[2][3] >= 24
        )
        if denominator_value == 6:
            denominator_ok = True
        elif structural_denominator_ok:
            denominator_ok = True
            denominator_value = 6
            denominator_confidence = max(float(denominator_confidence), 0.72)
        if denominator_value != 6:
            continue
        confidence = float((numerator_confidence + denominator_confidence) / 2.0)
        if confidence > best_confidence:
            best_value = numerator_value
            best_confidence = confidence
    return best_value, 6 if best_value is not None else None, best_confidence, denominator_ok


def _extract_mindscape(image: np.ndarray) -> tuple[int | None, int | None, float, list[str]]:
    reasons: list[str] = []
    region = _bottom_left_region(image)
    if region is None:
        return None, None, 0.0, ["agent_detail_mindscape_region_missing"]
    label_box = _detect_mindscape_label_box(region)
    if label_box is None:
        return None, None, 0.0, ["agent_detail_mindscape_label_missing"]
    crop = _extract_mindscape_numeric_crop(region, label_box)
    if crop is None:
        return None, None, 0.0, ["agent_detail_mindscape_crop_missing"]

    best_value, best_cap, best_confidence, denominator_ok = _parse_mindscape_numeric_crop(crop)
    if best_value is None:
        fixed_crop = _extract_mindscape_numeric_crop_fixed(region)
        if fixed_crop is not None:
            fixed_value, fixed_cap, fixed_confidence, fixed_denominator_ok = _parse_mindscape_numeric_crop(fixed_crop)
            if fixed_denominator_ok:
                denominator_ok = True
            if fixed_value is not None and fixed_confidence > best_confidence:
                best_value = fixed_value
                best_cap = fixed_cap
                best_confidence = fixed_confidence
    if best_value is None:
        reasons.append("agent_detail_mindscape_value_missing")
    if not denominator_ok:
        reasons.append("agent_detail_mindscape_denominator_invalid")
    return best_value, best_cap, best_confidence, reasons


def _extract_stat_value_block(
    image: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    field_name: str,
) -> np.ndarray | None:
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

    x0, y0, width0, height0 = components[-1]
    left = x0
    top = y0
    right = x0 + width0
    bottom = y0 + height0
    for x, y, width, height in reversed(components[:-1]):
        gap = left - (x + width)
        overlap = min(bottom, y + height) - max(top, y)
        if gap <= max(width, right - left) * 0.12 + 16 and overlap > min(height, bottom - top) * 0.35:
            left = x
            top = min(top, y)
            right = max(right, x + width)
            bottom = max(bottom, y + height)
            continue
        break

    pad_x = max(4, int(round((right - left) * 0.06)))
    pad_y = max(4, int(round((bottom - top) * 0.22)))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(row.shape[1], right + pad_x)
    bottom = min(row.shape[0], bottom + pad_y)
    block = row[top:bottom, left:right]
    return block if block.size else None


def _digit_tokens(block: np.ndarray) -> list[dict[str, Any]]:
    binary = _threshold_binary(block, 150, invert=False)
    components = _connected_components(binary, area_min=10, height_min=6, width_min=1)
    merged_boxes = _merge_overlapping_boxes(components)
    tokens: list[dict[str, Any]] = []
    for x0, y0, x1, y1 in merged_boxes:
        glyph = binary[y0:y1, x0:x1]
        label, score, margin = _classify_template_digit(glyph)
        if not label.isdigit():
            label, confidence = _classify_digit_component(glyph)
        else:
            confidence = _template_confidence(score, margin)
        tokens.append(
            {
                "x": int(x0),
                "y": int(y0),
                "width": int(x1 - x0),
                "height": int(y1 - y0),
                "digit": label,
                "confidence": confidence,
                "score": float(score),
                "margin": float(margin),
            }
        )
    return tokens


def _parse_stat_value(
    block: np.ndarray,
    kind: str,
) -> tuple[int | float | None, float, str | None]:
    tokens = _digit_tokens(block)
    if not tokens:
        return None, 0.0, "no_glyphs"

    median_height = statistics.median(token["height"] for token in tokens)
    median_width = statistics.median(token["width"] for token in tokens)
    median_y = statistics.median(token["y"] for token in tokens)

    typed_tokens: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        is_last = index == (len(tokens) - 1)
        is_separator = (
            token["height"] <= median_height * 0.68
            and token["width"] <= max(9.0, median_width * 0.62)
            and token["y"] >= median_y + median_height * 0.35
        )
        looks_percent = (
            kind == "percent"
            and is_last
            and token["height"] >= median_height * 0.82
            and token["width"] >= median_width * 1.12
        )
        typed = dict(token)
        if is_separator:
            typed["type"] = "sep"
        elif looks_percent:
            typed["type"] = "percent"
        else:
            typed["type"] = "digit"
        typed_tokens.append(typed)

    if kind == "int":
        typed_tokens = [token for token in typed_tokens if token["type"] != "sep"]

    digits = [str(token["digit"]) for token in typed_tokens if token["type"] == "digit"]
    if any(not digit.isdigit() for digit in digits):
        return None, 0.0, "nondigit"

    separator_count = sum(1 for token in typed_tokens if token["type"] == "sep")
    percent_count = sum(1 for token in typed_tokens if token["type"] == "percent")
    digit_confidences = [float(token["confidence"]) for token in typed_tokens if token["type"] == "digit"]
    confidence = float(sum(digit_confidences) / len(digit_confidences)) if digit_confidences else 0.0

    if kind == "int":
        if percent_count:
            return None, confidence, "bad_token_int"
        return int("".join(digits)), confidence, None

    if kind == "decimal":
        if percent_count or separator_count != 1 or len(digits) < 2:
            return None, confidence, "bad_token_decimal"
        return float(f"{''.join(digits[:-1])}.{digits[-1]}"), confidence, None

    if kind == "percent":
        if percent_count > 1 or separator_count > 1:
            return None, confidence, "bad_token_percent"
        if separator_count == 1:
            if len(digits) < 2:
                return None, confidence, "few_digits_percent"
            return float(f"{''.join(digits[:-1])}.{digits[-1]}"), confidence, None
        if not digits:
            return None, confidence, "no_digits_percent"
        return int("".join(digits)), confidence, None

    return None, confidence, "unsupported_kind"


def _extract_stats(image: np.ndarray) -> tuple[Dict[str, int | float], float, list[str]]:
    reasons: list[str] = []
    boxes = _detect_level_boxes(image)
    if boxes is None:
        return {}, 0.0, ["agent_detail_stats_boxes_missing"]

    parsed: Dict[str, int | float] = {}
    confidences: list[float] = []
    for spec in _STAT_SPECS:
        block = _extract_stat_value_block(image, boxes, str(spec["source"]))
        if block is None:
            reasons.append(f"agent_detail_stats_block_missing:{spec['canonical']}")
            continue
        value, confidence, error = _parse_stat_value(block, str(spec["kind"]))
        if error:
            reasons.append(f"agent_detail_stats_parse_failed:{spec['canonical']}:{error}")
            continue
        if value is None:
            reasons.append(f"agent_detail_stats_value_missing:{spec['canonical']}")
            continue
        numeric = float(value)
        if numeric < float(spec["min"]) or numeric > float(spec["max"]):
            reasons.append(f"agent_detail_stats_range_invalid:{spec['canonical']}:{value}")
            continue
        parsed[str(spec["canonical"])] = value
        confidences.append(confidence)

    parsed_count = len(parsed)
    if parsed_count < 4:
        reasons.append("agent_detail_stats_insufficient_fields")
        return {}, 0.0, reasons

    if parsed_count < len(_STAT_SPECS):
        reasons.append("agent_detail_stats_insufficient_fields")

    coverage_ratio = parsed_count / max(len(_STAT_SPECS), 1)
    confidence = (
        float(sum(confidences) / len(confidences)) * max(0.45, coverage_ratio)
        if confidences
        else 0.0
    )
    return parsed, confidence, reasons


def read_agent_detail(image: np.ndarray) -> AgentDetailReading:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    level, level_cap, level_confidence, level_reasons = _extract_level(gray)
    mindscape, mindscape_cap, mindscape_confidence, mindscape_reasons = _extract_mindscape(gray)
    stats, stats_confidence, stats_reasons = _extract_stats(gray)

    return AgentDetailReading(
        level=level,
        level_cap=level_cap,
        level_confidence=level_confidence,
        mindscape=mindscape,
        mindscape_cap=mindscape_cap,
        mindscape_confidence=mindscape_confidence,
        stats=stats,
        stats_confidence=stats_confidence,
        low_conf_reasons=level_reasons + mindscape_reasons + stats_reasons,
    )
