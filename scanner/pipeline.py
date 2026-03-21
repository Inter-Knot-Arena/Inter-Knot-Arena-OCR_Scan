from __future__ import annotations

import os
import random
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from amplifier_identity import (
    crop_effect_image,
    crop_info_image,
    crop_title_image,
    language_tag_for_locale,
    parse_amplifier_detail,
    run_winrt_ocr_batch,
)
from disc_identity import classify_disc_title
from .agent_detail_runtime import read_agent_detail
from .model_runtime import (
    ModelRegistry,
    classify_agent_icon,
    classify_disk_detail,
    classify_uid_digits,
    get_model_metadata,
    segment_uid_digits,
)
from .screen_runtime import (
    crop_agent_identity_candidates,
    derive_equipment_occupancy,
    normalize_runtime_captures,
    normalize_runtime_resolution,
)

SUPPORTED_LOCALES = {"RU", "EN"}
SUPPORTED_REGIONS = {"NA", "EU", "ASIA", "SEA", "OTHER"}
DEFAULT_MODEL_VERSION = "ocr-hybrid-v1.2"
_EQUIPMENT_WEAPON_CENTER = (0.694, 0.496)
_EQUIPMENT_WEAPON_PATCH = (0.120, 0.205)
_EQUIPMENT_DISC_SLOT_CENTERS = {
    1: (0.632, 0.284),
    2: (0.571, 0.487),
    3: (0.632, 0.691),
    4: (0.825, 0.691),
    5: (0.866, 0.487),
    6: (0.825, 0.284),
}
_EQUIPMENT_DISC_PATCH = (0.084, 0.138)
_EQUIPMENT_OCCUPIED_SCORE_THRESHOLD = 0.58
_EQUIPMENT_EMPTY_SCORE_THRESHOLD = 0.34
_AGENT_DETAIL_LEVEL_CONFIDENCE_THRESHOLD = 0.78


class ScanFailureCode(StrEnum):
    ANCHOR_MISMATCH = "ANCHOR_MISMATCH"
    UNSUPPORTED_LAYOUT = "UNSUPPORTED_LAYOUT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INPUT_LOCK_NOT_ACTIVE = "INPUT_LOCK_NOT_ACTIVE"


@dataclass(slots=True)
class ScanFailure(RuntimeError):
    code: ScanFailureCode
    message: str
    low_conf_reasons: List[str]
    partial_result: Dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def _normalize_locale(locale: str | None) -> str:
    value = (locale or "EN").strip().upper()
    return value if value in SUPPORTED_LOCALES else ""


def _normalize_region(region_hint: str | None) -> str:
    value = (region_hint or "OTHER").strip().upper()
    return value if value in SUPPORTED_REGIONS else "OTHER"


def _digits_only(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _runtime_temp_root(*segments: str) -> Path:
    override_root = str(os.environ.get("IKA_RUNTIME_TEMP_ROOT") or "").strip()
    base_root = Path(override_root) if override_root else Path(tempfile.gettempdir())
    root = base_root.joinpath(*segments)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_slot_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _visible_slot_key(page_index: Any, agent_slot_index: Any) -> tuple[int | None, int] | None:
    slot_index = _as_slot_index(agent_slot_index)
    if slot_index is None or slot_index <= 0:
        return None
    normalized_page_index = _as_slot_index(page_index)
    if normalized_page_index is not None and normalized_page_index < 0:
        normalized_page_index = None
    return normalized_page_index, slot_index


def _agent_visible_slot_key(agent: dict[str, Any], fallback_slot_index: int | None = None) -> tuple[int | None, int] | None:
    slot_key = _visible_slot_key(agent.get("_pageIndex"), agent.get("_agentSlotIndex"))
    if slot_key is not None:
        return slot_key
    return _visible_slot_key(None, fallback_slot_index)


def _copy_visible_slot_metadata(target: dict[str, Any], source: dict[str, Any]) -> None:
    for source_key, target_key in (
        ("pageIndex", "_pageIndex"),
        ("agentSlotIndex", "_agentSlotIndex"),
        ("_pageIndex", "_pageIndex"),
        ("_agentSlotIndex", "_agentSlotIndex"),
    ):
        value = source.get(source_key)
        normalized_index = _as_slot_index(value)
        if normalized_index is None:
            continue
        if target_key == "_pageIndex" and normalized_index < 0:
            continue
        if target_key == "_agentSlotIndex" and normalized_index <= 0:
            continue
        target[target_key] = normalized_index
    screen_alias = _as_text(source.get("screenAlias") or source.get("_screenAlias"))
    if screen_alias:
        target["_screenAlias"] = screen_alias


def _agent_ids_by_slot(agents: list[dict[str, Any]]) -> dict[tuple[int | None, int], str]:
    mapping: dict[tuple[int | None, int], str] = {}
    for index, agent in enumerate(agents, start=1):
        agent_id = _as_text(agent.get("agentId"))
        slot_key = _agent_visible_slot_key(agent, index)
        if agent_id and slot_key is not None:
            mapping[slot_key] = agent_id
    return mapping


def _merge_agent_ids_by_slot_from_details(
    slot_to_agent: dict[tuple[int | None, int], str],
    pixel_agent_details_by_slot: dict[tuple[int | None, int], dict[str, Any]],
) -> dict[tuple[int | None, int], str]:
    merged = dict(slot_to_agent)
    for slot_key, detail in pixel_agent_details_by_slot.items():
        if not isinstance(detail, dict):
            continue
        agent_id = _as_text(detail.get("agentId"))
        if slot_key is None or not agent_id:
            continue
        merged[slot_key] = agent_id
    return merged


def _lookup_agent_id_by_slot(
    slot_to_agent: dict[tuple[int | None, int], str],
    page_index: Any,
    agent_slot_index: Any,
) -> str:
    slot_key = _visible_slot_key(page_index, agent_slot_index)
    if slot_key is None:
        return ""
    direct = _as_text(slot_to_agent.get(slot_key))
    if direct:
        return direct
    fallback = _as_text(slot_to_agent.get((None, slot_key[1])))
    if fallback:
        return fallback
    candidate_ids = {
        agent_id
        for (candidate_page, candidate_slot), agent_id in slot_to_agent.items()
        if candidate_slot == slot_key[1]
        and (slot_key[0] is None or candidate_page is None)
        and _as_text(agent_id)
    }
    if len(candidate_ids) == 1:
        return next(iter(candidate_ids))
    return ""


def _classify_capture_agent_id(entry: dict[str, Any]) -> tuple[str, str, float | None, str]:
    if not ModelRegistry.has_agent_model():
        return "", "", None, "agent_model_missing_for_runtime_capture"

    role = _as_text(entry.get("role"))
    path_value = _as_text(entry.get("path"))
    if role not in {"agent_detail", "equipment", "amplifier_detail", "disk_detail"} or not path_value:
        return "", "", None, ""

    image = cv2.imread(str(Path(path_value)), cv2.IMREAD_COLOR)
    if image is None:
        return "", "", None, f"runtime_capture_agent_decode_failed:{role}"

    predictions: list[tuple[str, float]] = []
    for variant_name, crop in crop_agent_identity_candidates(image, role):
        prediction = classify_agent_icon(crop)
        predictions.append((prediction.label, float(prediction.confidence)))

    if len(predictions) < 2:
        return "", "", None, f"runtime_capture_agent_crop_missing:{role}"

    labels = {label for label, _ in predictions}
    if len(labels) != 1:
        labels_text = ",".join(sorted(labels))
        return "", "", None, f"runtime_capture_agent_disagree:{role}:{labels_text}"

    min_confidence = min(confidence for _, confidence in predictions)
    threshold = 0.78 if role == "agent_detail" else 0.82
    agreed_label = predictions[0][0]
    if min_confidence < threshold:
        return "", "", None, f"runtime_capture_agent_low_conf:{role}:{agreed_label}"

    source = f"{role}_portrait_agreement_onnx_agent_icon"
    return agreed_label, source, min_confidence, ""


def _resolve_capture_agent_id(
    entry: dict[str, Any],
    agent_ids_by_slot: dict[tuple[int | None, int], str],
    *,
    reasons: list[str] | None = None,
    reject_slot_mismatch: bool = False,
) -> tuple[str, str, float | None]:
    direct_agent_id = _as_text(entry.get("agentId") or entry.get("focusAgentId"))
    if direct_agent_id:
        return direct_agent_id, "screen_capture_agent_id", 0.99
    classified_agent_id, classified_source, classified_confidence, classified_reason = _classify_capture_agent_id(entry)
    agent_slot_index = _as_slot_index(entry.get("agentSlotIndex"))
    if agent_slot_index is not None:
        slot_agent_id = _lookup_agent_id_by_slot(
            agent_ids_by_slot,
            entry.get("pageIndex"),
            agent_slot_index,
        )
        if slot_agent_id and classified_agent_id:
            if slot_agent_id == classified_agent_id:
                return slot_agent_id, "visible_roster_slot_mapping_and_self_id", max(0.97, float(classified_confidence or 0.0))
            if float(classified_confidence or 0.0) >= 0.88:
                if reasons is not None:
                    reasons.append(
                        (
                            f"capture_agent_slot_mismatch_rejected:{agent_slot_index}:{slot_agent_id}->{classified_agent_id}"
                            if reject_slot_mismatch
                            else f"capture_agent_slot_override:{agent_slot_index}:{slot_agent_id}->{classified_agent_id}"
                        )
                    )
                if reject_slot_mismatch:
                    return "", "", None
                return classified_agent_id, classified_source, classified_confidence
            return slot_agent_id, "visible_roster_slot_mapping", 0.97
        if slot_agent_id:
            return slot_agent_id, "visible_roster_slot_mapping", 0.97

    if classified_reason and reasons is not None:
        reasons.append(classified_reason)
    if classified_agent_id:
        return classified_agent_id, classified_source, classified_confidence
    return "", "", None


def _select_uid_from_image(session_context: Dict[str, Any]) -> tuple[str | None, float | None, str | None, list[str]]:
    reasons: list[str] = []
    uid_path_raw = session_context.get("uidImagePath")
    if not isinstance(uid_path_raw, str) or not uid_path_raw.strip():
        return None, None, None, reasons
    uid_path = Path(uid_path_raw)
    if not uid_path.exists():
        reasons.append("uid_image_not_found")
        return None, None, None, reasons

    if not ModelRegistry.has_uid_model():
        reasons.append("uid_model_missing")
        return None, None, None, reasons

    image = cv2.imread(str(uid_path), cv2.IMREAD_COLOR)
    if image is None:
        reasons.append("uid_image_decode_failed")
        return None, None, None, reasons

    def try_uid_widget_ocr() -> tuple[str | None, float | None]:
        temp_root = _runtime_temp_root("ika_uid_widget_runtime", uid_path.stem)

        def run_ocr(candidate_path: Path) -> tuple[str | None, float | None]:
            ocr_text = _as_text(
                run_winrt_ocr_batch(
                    [{"id": "uid", "path": str(candidate_path)}],
                    language_tag=language_tag_for_locale("EN"),
                    temp_root=temp_root,
                ).get("uid")
            )
            digits = _digits_only(ocr_text)
            if 6 <= len(digits) <= 12:
                confidence = min(0.995, 0.90 + (len(digits) / 120.0))
                return digits, round(confidence, 4)
            return None, None

        direct_digits, direct_confidence = run_ocr(uid_path)
        if direct_digits:
            return direct_digits, direct_confidence

        if image.shape[0] <= 40 or image.shape[1] <= 180:
            for scale in (3, 4):
                resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                resized_path = temp_root / f"uid_x{scale}.png"
                if not cv2.imwrite(str(resized_path), resized):
                    continue
                scaled_digits, scaled_confidence = run_ocr(resized_path)
                if scaled_digits:
                    return scaled_digits, scaled_confidence
        return None, None

    digit_images = segment_uid_digits(image)
    if len(digit_images) < 6:
        fallback_uid, fallback_confidence = try_uid_widget_ocr()
        if fallback_uid:
            return fallback_uid, fallback_confidence, "uid_widget_winrt_ocr", reasons
        reasons.append("uid_digit_segmentation_failed")
        return None, None, None, reasons

    uid, confidence = classify_uid_digits(digit_images)
    digits_only = _digits_only(uid)
    if not (6 <= len(digits_only) <= 12):
        fallback_uid, fallback_confidence = try_uid_widget_ocr()
        if fallback_uid:
            return fallback_uid, fallback_confidence, "uid_widget_winrt_ocr", reasons
        reasons.append("uid_digit_length_invalid")
        return None, None, None, reasons
    return digits_only, confidence, "onnx_uid_digits", reasons


def _select_uid_from_candidates(candidates: Iterable[str]) -> tuple[str, float]:
    cleaned: list[str] = []
    for raw in candidates:
        value = _digits_only(str(raw))
        if 6 <= len(value) <= 12:
            cleaned.append(value)

    if cleaned:
        best = max(cleaned, key=len)
        confidence = min(0.998, 0.9 + (len(best) / 120))
        return best, round(confidence, 4)

    return "", 0.0


def _normalized_fraction(value: float, min_v: float, max_v: float) -> float:
    if value <= min_v:
        return 0.0
    if value >= max_v:
        return 1.0
    return (value - min_v) / (max_v - min_v)


def _has_meaningful_weapon_payload(weapon: Any) -> bool:
    if not isinstance(weapon, dict):
        return False
    return any(
        value not in (None, "", [], {})
        for key, value in weapon.items()
        if key not in {"agentId", "weaponPresent"}
    )


def _has_meaningful_disc_payload(discs: Any) -> bool:
    if not isinstance(discs, list):
        return False
    return any(
        isinstance(disc, dict)
        and (
            _as_slot_index(disc.get("slot")) is not None
            or _as_text(disc.get("setId"))
            or any(value not in (None, "", [], {}) for value in disc.values())
        )
        for disc in discs
    )


def _default_agent_payload(agent_id: str, confidence: float) -> dict[str, Any]:
    return {
        "agentId": agent_id,
        "level": None,
        "levelCap": None,
        "mindscape": None,
        "mindscapeCap": None,
        "stats": {},
        "weapon": {},
        "discs": [],
        "confidenceByField": {
            "agentId": round(confidence, 4),
            "level": 0.24,
            "mindscape": 0.24,
            "weapon": 0.18,
            "discs": 0.18,
            "occupancy": 0.16,
        },
        "fieldSources": {
            "agentId": "onnx_agent_icon",
            "level": "missing",
            "levelCap": "missing",
            "mindscape": "missing",
            "mindscapeCap": "missing",
            "stats": "missing",
            "weapon": "missing",
            "discs": "missing",
            "weaponPresent": "missing",
            "discSlotOccupancy": "missing",
        },
    }


def _known_empty_equipment_confidence(occupancy_confidence: float) -> float:
    return round(min(0.94, max(float(occupancy_confidence or 0.0), 0.90)), 4)


def _all_disc_slots_empty(slot_occupancy: Any) -> bool:
    return isinstance(slot_occupancy, dict) and bool(slot_occupancy) and not any(bool(value) for value in slot_occupancy.values())


def _disc_consensus_confidence(discs: Any) -> float:
    if not isinstance(discs, list):
        return 0.0

    slot_to_set: dict[int, str] = {}
    for disc in discs:
        if not isinstance(disc, dict):
            continue
        slot = _as_slot_index(disc.get("slot"))
        set_id = _as_text(disc.get("setId"))
        if slot is None or slot < 1 or slot > 6 or not set_id:
            continue
        slot_to_set[slot] = set_id

    if len(slot_to_set) != 6:
        return 0.0

    counts: dict[str, int] = {}
    for set_id in slot_to_set.values():
        counts[set_id] = counts.get(set_id, 0) + 1
    values = sorted(counts.values(), reverse=True)
    dominant = values[0]
    unique = len(values)

    if dominant == 6:
        return 0.92
    if dominant >= 5:
        return 0.88
    if dominant == 4 and unique <= 2:
        return 0.86
    if dominant == 4 and unique == 3:
        return 0.80
    if unique == 3 and values == [2, 2, 2]:
        return 0.80
    if unique == 2 and values == [3, 3]:
        return 0.80
    return 0.0


def _agents_from_icons(session_context: Dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    icon_payload = session_context.get("agentIconPaths")
    if not isinstance(icon_payload, list) or not icon_payload:
        return [], reasons

    if not ModelRegistry.has_agent_model():
        reasons.append("agent_model_missing")
        return [], reasons

    agents: list[dict[str, Any]] = []
    for index, entry in enumerate(icon_payload):
        path_value = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(path_value, str):
            reasons.append(f"agent_icon_invalid:{index}")
            continue
        path = Path(path_value)
        if not path.exists():
            reasons.append(f"agent_icon_not_found:{index}")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            reasons.append(f"agent_icon_decode_failed:{index}")
            continue
        prediction = classify_agent_icon(image)
        payload = _default_agent_payload(prediction.label, prediction.confidence)
        if isinstance(entry, dict):
            _copy_visible_slot_metadata(payload, entry)
        agents.append(payload)
        if prediction.confidence < 0.72:
            reasons.append(f"agent_icon_low_conf:{prediction.label}")

    return agents, reasons


def _pixel_discs_from_captures(
    session_context: Dict[str, Any],
    agent_ids_by_slot: dict[tuple[int | None, int], str],
    locale: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    reasons: list[str] = []
    raw_captures = session_context.get("screenCaptures")
    if not isinstance(raw_captures, list):
        return {}, reasons

    disk_entries = [
        entry
        for entry in raw_captures
        if isinstance(entry, dict) and _as_text(entry.get("role")) == "disk_detail" and _as_text(entry.get("path"))
    ]
    if not disk_entries:
        return {}, reasons

    if not ModelRegistry.has_disk_model():
        reasons.append("disk_model_missing")
        return {}, reasons

    by_agent_slot: dict[str, dict[int, dict[str, Any]]] = {}
    low_conf_requests: list[dict[str, Any]] = []
    for index, entry in enumerate(disk_entries):
        path_value = _as_text(entry.get("path"))
        agent_id, agent_source, agent_confidence = _resolve_capture_agent_id(
            entry,
            agent_ids_by_slot,
            reasons=reasons,
            reject_slot_mismatch=True,
        )
        slot_index = _as_slot_index(entry.get("slotIndex"))
        if not path_value:
            reasons.append(f"disk_detail_missing_path:{index}")
            continue
        if not agent_id:
            agent_slot_index = _as_slot_index(entry.get("agentSlotIndex"))
            reasons.append(
                f"disk_detail_missing_agent:{index}"
                if agent_slot_index is None
                else f"disk_detail_missing_agent_for_slot:{agent_slot_index}:{index}"
            )
            continue
        if slot_index is None or slot_index < 1 or slot_index > 6:
            reasons.append(f"disk_detail_missing_slot:{agent_id}:{index}")
            continue

        image = cv2.imread(str(Path(path_value)), cv2.IMREAD_COLOR)
        if image is None:
            reasons.append(f"disk_detail_decode_failed:{agent_id}:{slot_index}")
            continue

        prediction = classify_disk_detail(image)

        agent_slots = by_agent_slot.setdefault(agent_id, {})
        candidate = {
            "slot": slot_index,
            "setId": prediction.label,
            "agentId": agent_id,
            "agentSource": agent_source,
            "agentConfidence": float(agent_confidence or 0.0),
            "_confidence": float(prediction.confidence),
            "_source": "onnx_disk_detail",
        }
        if float(prediction.confidence) < 0.72:
            low_conf_requests.append(
                {
                    "captureId": f"disk_{index}_{agent_id}_{slot_index}",
                    "agentId": agent_id,
                    "slotIndex": slot_index,
                    "path": path_value,
                    "onnxLabel": prediction.label,
                }
            )
        existing = agent_slots.get(slot_index)
        if existing is None or float(candidate["_confidence"]) > float(existing["_confidence"]):
            agent_slots[slot_index] = candidate

    if low_conf_requests:
        language_tag = language_tag_for_locale(locale)
        temp_base = _runtime_temp_root("disk_runtime")
        with tempfile.TemporaryDirectory(prefix="disk_runtime_", dir=str(temp_base)) as raw_tmp:
            temp_root = Path(raw_tmp)
            crop_dir = temp_root / "crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crops: list[dict[str, str]] = []
            for request in low_conf_requests:
                source_path = Path(_as_text(request.get("path")))
                capture_id = _as_text(request.get("captureId"))
                try:
                    title_crop_path = crop_dir / f"{capture_id}_title.png"
                    crop_title_image(source_path, title_crop_path)
                except Exception:
                    continue
                crops.append({"id": f"{capture_id}:title", "path": str(title_crop_path)})

            ocr_results = run_winrt_ocr_batch(crops, language_tag=language_tag, temp_root=temp_root) if crops else {}
            for request in low_conf_requests:
                agent_id = _as_text(request.get("agentId"))
                slot_index = _as_slot_index(request.get("slotIndex"))
                capture_id = _as_text(request.get("captureId"))
                if not agent_id or slot_index is None:
                    continue
                slot_map = by_agent_slot.get(agent_id)
                candidate = slot_map.get(slot_index) if isinstance(slot_map, dict) else None
                if not isinstance(candidate, dict):
                    continue
                title_prediction = classify_disc_title(_as_text(ocr_results.get(f"{capture_id}:title")))
                if title_prediction is None or float(title_prediction.confidence) <= float(candidate.get("_confidence", 0.0)):
                    continue
                candidate["setId"] = title_prediction.set_id
                candidate["displayName"] = title_prediction.display_name
                candidate["_confidence"] = float(title_prediction.confidence)
                candidate["_source"] = str(title_prediction.source_mode)

    for agent_id, slot_map in by_agent_slot.items():
        for slot_index, candidate in slot_map.items():
            if float(candidate.get("_confidence", 0.0) or 0.0) < 0.72:
                reasons.append(f"disk_detail_low_conf:{agent_id}:{slot_index}:{_as_text(candidate.get('setId'))}")

    output: dict[str, list[dict[str, Any]]] = {}
    for agent_id, slot_map in by_agent_slot.items():
        output[agent_id] = [slot_map[slot] for slot in sorted(slot_map)]
    return output, reasons


def _pixel_agent_details_from_captures(
    session_context: Dict[str, Any],
    agent_ids_by_slot: dict[tuple[int | None, int], str],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[int | None, int], dict[str, Any]], list[str]]:
    reasons: list[str] = []
    raw_captures = session_context.get("screenCaptures")
    if not isinstance(raw_captures, list):
        return {}, {}, reasons

    if not ModelRegistry.has_uid_model():
        reasons.append("uid_model_missing_for_agent_detail")
        return {}, {}, reasons

    detail_entries = [
        entry
        for entry in raw_captures
        if isinstance(entry, dict) and _as_text(entry.get("role")) == "agent_detail" and _as_text(entry.get("path"))
    ]
    if not detail_entries:
        return {}, {}, reasons

    by_agent: dict[str, dict[str, Any]] = {}
    by_slot: dict[tuple[int | None, int], dict[str, Any]] = {}
    for index, entry in enumerate(detail_entries):
        path_value = _as_text(entry.get("path"))
        agent_id, agent_source, agent_confidence = _resolve_capture_agent_id(
            entry,
            agent_ids_by_slot,
            reasons=reasons,
            reject_slot_mismatch=True,
        )
        if not path_value:
            reasons.append(f"agent_detail_missing_path:{index}")
            continue
        if not agent_id:
            agent_slot_index = _as_slot_index(entry.get("agentSlotIndex"))
            reasons.append(
                f"agent_detail_missing_agent:{index}"
                if agent_slot_index is None
                else f"agent_detail_missing_agent_for_slot:{agent_slot_index}:{index}"
            )
            continue

        image = cv2.imread(str(Path(path_value)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            reasons.append(f"agent_detail_decode_failed:{agent_id}:{index}")
            continue

        reading = read_agent_detail(image)
        candidate = {
            "agentId": agent_id,
            "agentSource": agent_source,
            "agentConfidence": float(agent_confidence or 0.0),
            "pageIndex": _as_slot_index(entry.get("pageIndex")),
            "agentSlotIndex": _as_slot_index(entry.get("agentSlotIndex")),
            "level": reading.level,
            "levelCap": reading.level_cap,
            "levelConfidence": float(reading.level_confidence),
            "mindscape": reading.mindscape,
            "mindscapeCap": reading.mindscape_cap,
            "mindscapeConfidence": float(reading.mindscape_confidence),
            "stats": dict(reading.stats),
            "statsConfidence": float(reading.stats_confidence),
            "lowConfReasons": list(reading.low_conf_reasons),
        }
        if (
            reading.level is None
            or reading.level_cap is None
            or reading.level_confidence < _AGENT_DETAIL_LEVEL_CONFIDENCE_THRESHOLD
        ):
            reasons.append(f"agent_detail_level_low_conf:{agent_id}")
        if reading.mindscape is None or reading.mindscape_cap is None or reading.mindscape_confidence < 0.50:
            reasons.append(f"agent_detail_mindscape_low_conf:{agent_id}")
        reasons.extend(
            f"{agent_id}:{value}"
            for value in reading.low_conf_reasons
            if not str(value).startswith("agent_detail_stats_")
        )

        candidate_score = (
            float(candidate["levelConfidence"])
            + float(candidate["mindscapeConfidence"])
            + float(candidate["statsConfidence"])
        )
        slot_key = _visible_slot_key(candidate.get("pageIndex"), candidate.get("agentSlotIndex"))
        if slot_key is not None:
            existing_for_slot = by_slot.get(slot_key)
            existing_slot_score = (
                float(existing_for_slot["levelConfidence"])
                + float(existing_for_slot["mindscapeConfidence"])
                + float(existing_for_slot["statsConfidence"])
            ) if isinstance(existing_for_slot, dict) else -1.0
            if existing_for_slot is None or candidate_score > existing_slot_score:
                by_slot[slot_key] = candidate

        existing = by_agent.get(agent_id)
        if existing is None:
            by_agent[agent_id] = candidate
            continue

        existing_score = (
            float(existing["levelConfidence"])
            + float(existing["mindscapeConfidence"])
            + float(existing["statsConfidence"])
        )
        if candidate_score > existing_score:
            by_agent[agent_id] = candidate

    return by_agent, by_slot, reasons


def _fractional_patch_from_center(
    image: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    width_fraction: float,
    height_fraction: float,
) -> np.ndarray | None:
    height, width = image.shape[:2]
    patch_width = max(1, int(round(width * width_fraction)))
    patch_height = max(1, int(round(height * height_fraction)))
    center_px = int(round(width * center_x))
    center_py = int(round(height * center_y))
    x0 = max(0, center_px - patch_width // 2)
    y0 = max(0, center_py - patch_height // 2)
    x1 = min(width, x0 + patch_width)
    y1 = min(height, y0 + patch_height)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def _texture_score_for_occupancy(patch: np.ndarray) -> float:
    if patch.ndim == 3:
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = patch
    if gray.size == 0:
        return 0.0

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    contrast = min(1.0, float(np.std(blurred)) / 52.0)
    laplacian_var = min(1.0, float(cv2.Laplacian(blurred, cv2.CV_32F).var()) / 900.0)
    edges = cv2.Canny(blurred, 60, 140)
    edge_density = min(1.0, float((edges > 0).mean()) / 0.12)

    inner_margin_y = max(1, blurred.shape[0] // 5)
    inner_margin_x = max(1, blurred.shape[1] // 5)
    inner = blurred[inner_margin_y:-inner_margin_y, inner_margin_x:-inner_margin_x]
    outer_mask = np.ones_like(blurred, dtype=bool)
    if inner.size:
        outer_mask[inner_margin_y:-inner_margin_y, inner_margin_x:-inner_margin_x] = False
    outer = blurred[outer_mask]
    inner_std = float(np.std(inner)) if inner.size else 0.0
    outer_std = float(np.std(outer)) if outer.size else 0.0
    structure_gain = min(1.0, max(0.0, inner_std - outer_std) / 28.0)

    return round(
        (0.30 * contrast)
        + (0.35 * laplacian_var)
        + (0.20 * edge_density)
        + (0.15 * structure_gain),
        4,
    )


def _presence_from_patch(
    image: np.ndarray,
    *,
    center: tuple[float, float],
    patch_size: tuple[float, float],
) -> tuple[bool | None, float]:
    patch = _fractional_patch_from_center(
        image,
        center_x=float(center[0]),
        center_y=float(center[1]),
        width_fraction=float(patch_size[0]),
        height_fraction=float(patch_size[1]),
    )
    if patch is None:
        return None, 0.0
    score = _texture_score_for_occupancy(patch)
    if score >= _EQUIPMENT_OCCUPIED_SCORE_THRESHOLD:
        return True, score
    if score <= _EQUIPMENT_EMPTY_SCORE_THRESHOLD:
        return False, 1.0 - score
    return None, score


def _derive_equipment_overview_occupancy_from_image(
    image: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    occupancy: dict[str, Any] = {}

    weapon_present, weapon_confidence = _presence_from_patch(
        image,
        center=_EQUIPMENT_WEAPON_CENTER,
        patch_size=_EQUIPMENT_WEAPON_PATCH,
    )
    if weapon_present is None:
        reasons.append("equipment_overview_weapon_presence_ambiguous")
    else:
        occupancy["weaponPresent"] = weapon_present
        occupancy["_weaponConfidence"] = round(float(weapon_confidence), 4)

    disc_slot_occupancy: dict[str, bool] = {}
    disc_confidences: list[float] = []
    for slot_index, center in sorted(_EQUIPMENT_DISC_SLOT_CENTERS.items()):
        present, confidence = _presence_from_patch(
            image,
            center=center,
            patch_size=_EQUIPMENT_DISC_PATCH,
        )
        if present is None:
            reasons.append(f"equipment_overview_slot_ambiguous:{slot_index}")
            disc_slot_occupancy = {}
            disc_confidences = []
            break
        disc_slot_occupancy[str(slot_index)] = present
        disc_confidences.append(float(confidence))
    if disc_slot_occupancy:
        occupancy["discSlotOccupancy"] = disc_slot_occupancy
        occupancy["_discConfidence"] = round(sum(disc_confidences) / len(disc_confidences), 4)

    return occupancy, reasons


def _pixel_equipment_occupancy_from_captures(
    session_context: Dict[str, Any],
    agent_ids_by_slot: dict[tuple[int | None, int], str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    reasons: list[str] = []
    raw_captures = session_context.get("screenCaptures")
    if not isinstance(raw_captures, list):
        return {}, reasons

    equipment_entries = [
        entry
        for entry in raw_captures
        if isinstance(entry, dict) and _as_text(entry.get("role")) == "equipment" and _as_text(entry.get("path"))
    ]
    if not equipment_entries:
        return {}, reasons

    by_agent: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(equipment_entries):
        path_value = _as_text(entry.get("path"))
        agent_id, agent_source, agent_confidence = _resolve_capture_agent_id(
            entry,
            agent_ids_by_slot,
            reasons=reasons,
            reject_slot_mismatch=True,
        )
        if not path_value:
            reasons.append(f"equipment_overview_missing_path:{index}")
            continue
        if not agent_id:
            agent_slot_index = _as_slot_index(entry.get("agentSlotIndex"))
            reasons.append(
                f"equipment_overview_missing_agent:{index}"
                if agent_slot_index is None
                else f"equipment_overview_missing_agent_for_slot:{agent_slot_index}:{index}"
            )
            continue

        image = cv2.imread(str(Path(path_value)), cv2.IMREAD_COLOR)
        if image is None:
            reasons.append(f"equipment_overview_decode_failed:{agent_id}:{index}")
            continue

        occupancy, occupancy_reasons = _derive_equipment_overview_occupancy_from_image(image)
        reasons.extend(f"{agent_id}:{reason}" for reason in occupancy_reasons)
        if "weaponPresent" not in occupancy and "discSlotOccupancy" not in occupancy:
            continue

        candidate = {
            "agentId": agent_id,
            "agentSource": agent_source,
            "agentConfidence": float(agent_confidence or 0.0),
            "pageIndex": _as_slot_index(entry.get("pageIndex")),
            "agentSlotIndex": _as_slot_index(entry.get("agentSlotIndex")),
            "_confidence": max(
                float(occupancy.get("_weaponConfidence", 0.0) or 0.0),
                float(occupancy.get("_discConfidence", 0.0) or 0.0),
            ),
        }
        if "weaponPresent" in occupancy:
            candidate["weaponPresent"] = bool(occupancy["weaponPresent"])
            candidate["_weaponConfidence"] = float(occupancy.get("_weaponConfidence", 0.0) or 0.0)
        if "discSlotOccupancy" in occupancy:
            candidate["discSlotOccupancy"] = dict(occupancy["discSlotOccupancy"])
            candidate["_discConfidence"] = float(occupancy.get("_discConfidence", 0.0) or 0.0)

        existing = by_agent.get(agent_id)
        if existing is None or float(candidate["_confidence"]) > float(existing.get("_confidence", 0.0)):
            by_agent[agent_id] = candidate

    return by_agent, reasons


def _pixel_weapons_from_captures(
    session_context: Dict[str, Any],
    agent_ids_by_slot: dict[tuple[int | None, int], str],
    occupancy_by_agent: dict[str, dict[str, Any]],
    locale: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    reasons: list[str] = []
    raw_captures = session_context.get("screenCaptures")
    if not isinstance(raw_captures, list):
        return {}, reasons

    amplifier_entries = [
        entry
        for entry in raw_captures
        if isinstance(entry, dict) and _as_text(entry.get("role")) == "amplifier_detail" and _as_text(entry.get("path"))
    ]
    if not amplifier_entries:
        return {}, reasons

    by_capture_id: dict[str, dict[str, Any]] = {}
    ocr_results: dict[str, str] = {}
    language_tag = language_tag_for_locale(locale)

    temp_base = _runtime_temp_root("amp_runtime")
    with tempfile.TemporaryDirectory(prefix="amp_runtime_", dir=str(temp_base)) as raw_tmp:
        temp_root = Path(raw_tmp)
        crop_dir = temp_root / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crops: list[dict[str, str]] = []

        for index, entry in enumerate(amplifier_entries):
            path_value = _as_text(entry.get("path"))
            agent_id, agent_source, agent_confidence = _resolve_capture_agent_id(
                entry,
                agent_ids_by_slot,
                reasons=reasons,
            )
            if not path_value:
                reasons.append(f"amplifier_detail_missing_path:{index}")
                continue
            if not agent_id:
                agent_slot_index = _as_slot_index(entry.get("agentSlotIndex"))
                reasons.append(
                    f"amplifier_detail_missing_agent:{index}"
                    if agent_slot_index is None
                    else f"amplifier_detail_missing_agent_for_slot:{agent_slot_index}:{index}"
                )
                continue
            occupancy = occupancy_by_agent.get(agent_id)
            if isinstance(occupancy, dict) and occupancy.get("weaponPresent") is False:
                continue
            source_path = Path(path_value)
            if not source_path.exists():
                reasons.append(f"amplifier_detail_not_found:{agent_id}:{index}")
                continue
            capture_id = f"amplifier_{index}_{agent_id}"
            try:
                title_crop_path = crop_dir / f"{capture_id}_title.png"
                crop_title_image(source_path, title_crop_path)
            except Exception:
                reasons.append(f"amplifier_detail_crop_failed:{agent_id}:{index}")
                continue
            crops.append({"id": f"{capture_id}:title", "path": str(title_crop_path)})
            try:
                info_crop_path = crop_dir / f"{capture_id}_info.png"
                crop_info_image(source_path, info_crop_path)
                crops.append({"id": f"{capture_id}:info", "path": str(info_crop_path)})
            except Exception:
                reasons.append(f"amplifier_detail_info_crop_failed:{agent_id}:{index}")
            try:
                effect_crop_path = crop_dir / f"{capture_id}_effect.png"
                crop_effect_image(source_path, effect_crop_path)
                crops.append({"id": f"{capture_id}:effect", "path": str(effect_crop_path)})
            except Exception:
                reasons.append(f"amplifier_detail_effect_crop_failed:{agent_id}:{index}")
            by_capture_id[capture_id] = {
                "agentId": agent_id,
                "agentSource": agent_source,
                "agentConfidence": float(agent_confidence or 0.0),
                "index": index,
            }

        if crops:
            ocr_results.update(run_winrt_ocr_batch(crops, language_tag=language_tag, temp_root=temp_root))

    by_agent: dict[str, dict[str, Any]] = {}
    for capture_id, meta in by_capture_id.items():
        agent_id = _as_text(meta.get("agentId"))
        title_text = _as_text(ocr_results.get(f"{capture_id}:title"))
        info_text = _as_text(ocr_results.get(f"{capture_id}:info"))
        effect_text = _as_text(ocr_results.get(f"{capture_id}:effect"))
        readout = parse_amplifier_detail(title_text, info_text=info_text, effect_text=effect_text)
        if readout is None:
            reasons.append(f"amplifier_detail_unclassified:{agent_id}")
            continue
        prediction = readout.identity
        if prediction.source_mode not in {"alias_contract", "alias_contract_fuzzy"}:
            reasons.append(f"amplifier_detail_low_conf:{agent_id}:{prediction.source_mode}:{prediction.weapon_id}")
            continue
        candidate = {
            "agentId": agent_id,
            "agentSource": _as_text(meta.get("agentSource")),
            "agentConfidence": float(meta.get("agentConfidence", 0.0) or 0.0),
            "weaponId": prediction.weapon_id,
            "displayName": prediction.display_name,
            "weaponPresent": True,
            "_confidence": float(prediction.confidence),
            "_source": str(prediction.source_mode),
        }
        if readout.level is not None:
            candidate["level"] = int(readout.level)
        if readout.level_cap is not None:
            candidate["levelCap"] = int(readout.level_cap)
        if readout.base_stat_key:
            candidate["baseStatKey"] = readout.base_stat_key
        if readout.base_stat_value not in (None, ""):
            candidate["baseStatValue"] = readout.base_stat_value
        if readout.advanced_stat_key:
            candidate["advancedStatKey"] = readout.advanced_stat_key
        if readout.advanced_stat_value not in (None, ""):
            candidate["advancedStatValue"] = readout.advanced_stat_value
        if readout.level is None or readout.level_cap is None:
            reasons.append(f"amplifier_detail_level_missing:{agent_id}")
        if readout.base_stat_key is None or readout.base_stat_value in (None, ""):
            reasons.append(f"amplifier_detail_base_stat_missing:{agent_id}")
        if readout.advanced_stat_key is None or readout.advanced_stat_value in (None, ""):
            reasons.append(f"amplifier_detail_advanced_stat_missing:{agent_id}")
        existing = by_agent.get(agent_id)
        if existing is None or float(candidate["_confidence"]) > float(existing.get("_confidence", 0.0)):
            by_agent[agent_id] = candidate

    return by_agent, reasons


def _merge_agents(
    parsed_agents: list[dict[str, Any]],
    icon_agents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    reasons: list[str] = []

    def dedupe_agents(
        agents: list[dict[str, Any]],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for agent in agents:
            agent_id = _as_text(agent.get("agentId"))
            if not agent_id:
                deduped.append(agent)
                continue
            if agent_id in seen_ids:
                reasons.append(f"duplicate_agent_id_from_{source}:{agent_id}")
                continue
            seen_ids.add(agent_id)
            deduped.append(agent)
        return deduped

    parsed_agents = dedupe_agents(parsed_agents, source="session_payload")
    icon_agents = dedupe_agents(icon_agents, source="onnx_agent_icon")

    if not parsed_agents:
        return icon_agents, reasons
    if not icon_agents:
        return parsed_agents, reasons

    merged: list[dict[str, Any]] = []
    parsed_by_id = {
        _as_text(agent.get("agentId")): agent
        for agent in parsed_agents
        if _as_text(agent.get("agentId"))
    }
    used_parsed_ids: set[str] = set()

    for icon in icon_agents:
        icon_agent_id = _as_text(icon.get("agentId"))
        parsed = parsed_by_id.get(icon_agent_id)
        if parsed is None:
            if icon_agent_id and parsed_by_id:
                reasons.append(f"agent_icon_without_payload:{icon_agent_id}")
            merged.append(icon)
            continue
        merged_payload = dict(parsed)
        merged_confidence = dict(parsed.get("confidenceByField") or {})
        merged_field_sources = dict(parsed.get("fieldSources") or {})
        icon_confidence = icon.get("confidenceByField")
        _copy_visible_slot_metadata(merged_payload, icon)
        if icon_agent_id:
            merged_payload["agentId"] = icon_agent_id
            merged_field_sources["agentId"] = "onnx_agent_icon"
            used_parsed_ids.add(icon_agent_id)
        if isinstance(icon_confidence, dict):
            agent_conf = icon_confidence.get("agentId")
            if isinstance(agent_conf, (int, float)):
                merged_confidence["agentId"] = round(float(agent_conf), 4)
        merged_payload["confidenceByField"] = merged_confidence
        merged_payload["fieldSources"] = merged_field_sources
        merged.append(merged_payload)

    for parsed in parsed_agents:
        parsed_agent_id = _as_text(parsed.get("agentId"))
        if parsed_agent_id and parsed_agent_id in used_parsed_ids:
            continue
        if parsed_agent_id:
            reasons.append(f"agent_payload_not_visible_in_roster:{parsed_agent_id}")
        merged.append(parsed)

    return merged, reasons


def _enrich_agents_with_pixel_discs(
    agents: list[dict[str, Any]],
    pixel_discs_by_agent: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], bool]:
    if not pixel_discs_by_agent:
        return agents, False

    merged_agents: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    used = False

    for agent in agents:
        agent_id = _as_text(agent.get("agentId"))
        seen_agents.add(agent_id)
        pixel_discs = list(pixel_discs_by_agent.get(agent_id) or [])
        if not pixel_discs:
            merged_agents.append(agent)
            continue

        payload = dict(agent)
        existing_discs_raw = payload.get("discs")
        existing_discs = list(existing_discs_raw) if isinstance(existing_discs_raw, list) else []
        existing_slots = {
            _as_slot_index(disc.get("slot")) for disc in existing_discs if isinstance(disc, dict)
        }
        appended = [
            {key: value for key, value in disc.items() if not str(key).startswith("_")}
            for disc in pixel_discs
            if _as_slot_index(disc.get("slot")) not in existing_slots
        ]
        if not appended:
            merged_agents.append(agent)
            continue

        payload["discs"] = sorted(existing_discs + appended, key=lambda disc: _as_slot_index(disc.get("slot")) or 0)
        confidence_by_field = dict(payload.get("confidenceByField") or {})
        field_sources = dict(payload.get("fieldSources") or {})
        avg_confidence = sum(float(disc.get("_confidence", 0.0)) for disc in pixel_discs) / max(len(pixel_discs), 1)
        consensus_confidence = _disc_consensus_confidence(payload.get("discs"))
        effective_confidence = max(float(avg_confidence), float(consensus_confidence))
        previous_disc_source = _as_text(field_sources.get("discs"))
        confidence_by_field["discs"] = round(max(float(confidence_by_field.get("discs", 0.0)), effective_confidence), 4)
        confidence_by_field["occupancy"] = round(max(float(confidence_by_field.get("occupancy", 0.0)), effective_confidence), 4)
        if previous_disc_source in {"", "missing"}:
            field_sources["discs"] = "onnx_disk_detail"
            field_sources["discSlotOccupancy"] = "derived_from_onnx_disk_detail"
        else:
            field_sources["discs"] = "merged_session_payload_and_onnx_disk_detail"
            field_sources["discSlotOccupancy"] = "merged_session_payload_and_onnx_disk_detail"
        occupancy = derive_equipment_occupancy(
            weapon=payload.get("weapon") if isinstance(payload.get("weapon"), dict) else None,
            discs=payload.get("discs") if isinstance(payload.get("discs"), list) else None,
            explicit_weapon_present=payload.get("weaponPresent"),
            explicit_slot_occupancy=payload.get("discSlotOccupancy"),
        )
        payload["weaponPresent"] = occupancy["weaponPresent"]
        payload["discSlotOccupancy"] = occupancy["discSlotOccupancy"]
        payload["confidenceByField"] = confidence_by_field
        payload["fieldSources"] = field_sources
        merged_agents.append(payload)
        used = True

    for agent_id, pixel_discs in sorted(pixel_discs_by_agent.items()):
        if agent_id in seen_agents:
            continue
        avg_confidence = sum(float(disc.get("_confidence", 0.0)) for disc in pixel_discs) / max(len(pixel_discs), 1)
        agent_confidence = max(float(disc.get("agentConfidence", 0.0) or 0.0) for disc in pixel_discs)
        agent_source = next(
            (str(disc.get("agentSource") or "").strip() for disc in pixel_discs if str(disc.get("agentSource") or "").strip()),
            "screen_capture_agent_id",
        )
        payload = _default_agent_payload(agent_id, max(avg_confidence, agent_confidence or 0.96))
        payload["fieldSources"]["agentId"] = agent_source
        payload["confidenceByField"]["agentId"] = round(
            max(float(payload["confidenceByField"]["agentId"]), agent_confidence or 0.96),
            4,
        )
        payload["discs"] = [
            {key: value for key, value in disc.items() if not str(key).startswith("_")}
            for disc in pixel_discs
        ]
        consensus_confidence = _disc_consensus_confidence(payload.get("discs"))
        effective_confidence = max(float(avg_confidence), float(consensus_confidence))
        payload["confidenceByField"]["discs"] = round(effective_confidence, 4)
        payload["confidenceByField"]["occupancy"] = round(effective_confidence, 4)
        payload["fieldSources"]["discs"] = "onnx_disk_detail"
        payload["fieldSources"]["discSlotOccupancy"] = "derived_from_onnx_disk_detail"
        occupancy = derive_equipment_occupancy(
            weapon=payload.get("weapon") if isinstance(payload.get("weapon"), dict) else None,
            discs=payload.get("discs") if isinstance(payload.get("discs"), list) else None,
            explicit_weapon_present=payload.get("weaponPresent"),
            explicit_slot_occupancy=payload.get("discSlotOccupancy"),
        )
        payload["weaponPresent"] = occupancy["weaponPresent"]
        payload["discSlotOccupancy"] = occupancy["discSlotOccupancy"]
        merged_agents.append(payload)
        used = True

    return merged_agents, used


def _enrich_agents_with_pixel_equipment_occupancy(
    agents: list[dict[str, Any]],
    pixel_equipment_occupancy_by_agent: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if not pixel_equipment_occupancy_by_agent:
        return agents, False

    merged_agents: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    used = False

    for agent in agents:
        payload = dict(agent)
        agent_id = _as_text(payload.get("agentId"))
        seen_agents.add(agent_id)
        occupancy = pixel_equipment_occupancy_by_agent.get(agent_id)
        if not occupancy:
            merged_agents.append(payload)
            continue

        confidence_by_field = dict(payload.get("confidenceByField") or {})
        field_sources = dict(payload.get("fieldSources") or {})
        _copy_visible_slot_metadata(payload, occupancy)

        if "weaponPresent" in occupancy:
            payload["weaponPresent"] = bool(occupancy["weaponPresent"])
            confidence_by_field["occupancy"] = round(
                max(float(confidence_by_field.get("occupancy", 0.0)), float(occupancy.get("_weaponConfidence", 0.0) or 0.0)),
                4,
            )
            field_sources["weaponPresent"] = "pixel_equipment_overview"
            if not payload.get("weapon") and payload["weaponPresent"] is False:
                known_empty_confidence = _known_empty_equipment_confidence(float(occupancy.get("_weaponConfidence", 0.0) or 0.0))
                confidence_by_field["weapon"] = round(
                    max(float(confidence_by_field.get("weapon", 0.0)), known_empty_confidence),
                    4,
                )
                field_sources["weapon"] = "known_empty_from_equipment_occupancy"
            used = True
        if "discSlotOccupancy" in occupancy:
            payload["discSlotOccupancy"] = dict(occupancy["discSlotOccupancy"])
            confidence_by_field["occupancy"] = round(
                max(float(confidence_by_field.get("occupancy", 0.0)), float(occupancy.get("_discConfidence", 0.0) or 0.0)),
                4,
            )
            field_sources["discSlotOccupancy"] = "pixel_equipment_overview"
            if not payload.get("discs") and _all_disc_slots_empty(payload["discSlotOccupancy"]):
                known_empty_confidence = _known_empty_equipment_confidence(float(occupancy.get("_discConfidence", 0.0) or 0.0))
                confidence_by_field["discs"] = round(
                    max(float(confidence_by_field.get("discs", 0.0)), known_empty_confidence),
                    4,
                )
                field_sources["discs"] = "known_empty_from_equipment_occupancy"
            used = True

        payload["confidenceByField"] = confidence_by_field
        payload["fieldSources"] = field_sources
        merged_agents.append(payload)

    for agent_id, occupancy in sorted(pixel_equipment_occupancy_by_agent.items()):
        if agent_id in seen_agents:
            continue
        agent_source = _as_text(occupancy.get("agentSource")) or "screen_capture_agent_id"
        agent_confidence = float(occupancy.get("agentConfidence", 0.0) or 0.0)
        payload = _default_agent_payload(agent_id, max(agent_confidence, 0.96))
        payload["fieldSources"]["agentId"] = agent_source
        payload["confidenceByField"]["agentId"] = round(
            max(float(payload["confidenceByField"]["agentId"]), agent_confidence or 0.96),
            4,
        )
        _copy_visible_slot_metadata(payload, occupancy)
        if "weaponPresent" in occupancy:
            payload["weaponPresent"] = bool(occupancy["weaponPresent"])
            payload["fieldSources"]["weaponPresent"] = "pixel_equipment_overview"
            if payload["weaponPresent"] is False:
                payload["confidenceByField"]["weapon"] = _known_empty_equipment_confidence(float(occupancy.get("_weaponConfidence", 0.0) or 0.0))
                payload["fieldSources"]["weapon"] = "known_empty_from_equipment_occupancy"
        if "discSlotOccupancy" in occupancy:
            payload["discSlotOccupancy"] = dict(occupancy["discSlotOccupancy"])
            payload["fieldSources"]["discSlotOccupancy"] = "pixel_equipment_overview"
            if _all_disc_slots_empty(payload["discSlotOccupancy"]):
                payload["confidenceByField"]["discs"] = _known_empty_equipment_confidence(float(occupancy.get("_discConfidence", 0.0) or 0.0))
                payload["fieldSources"]["discs"] = "known_empty_from_equipment_occupancy"
        payload["confidenceByField"]["occupancy"] = round(float(occupancy.get("_confidence", 0.0) or 0.0), 4)
        merged_agents.append(payload)
        used = True

    return merged_agents, used


def _enrich_agents_with_pixel_weapons(
    agents: list[dict[str, Any]],
    pixel_weapons_by_agent: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if not pixel_weapons_by_agent:
        return agents, False

    def _weapon_source(pixel_weapon: dict[str, Any]) -> str:
        if any(
            pixel_weapon.get(key) not in (None, "", [], {})
            for key in ("level", "levelCap", "baseStatKey", "baseStatValue", "advancedStatKey", "advancedStatValue")
        ):
            return "amplifier_detail_ocr"
        return "amplifier_detail_title_ocr"

    def _merge_weapon_fields(
        current_weapon: dict[str, Any],
        pixel_weapon: dict[str, Any],
        *,
        agent_id: str,
        allow_identity_fill: bool,
    ) -> tuple[dict[str, Any], bool]:
        merged_weapon = dict(current_weapon)
        changed = False
        pixel_weapon_id = _as_text(pixel_weapon.get("weaponId"))
        existing_weapon_id = _as_text(merged_weapon.get("weaponId"))

        if allow_identity_fill and pixel_weapon_id and not existing_weapon_id:
            merged_weapon["weaponId"] = pixel_weapon_id
            changed = True
        elif existing_weapon_id and pixel_weapon_id and existing_weapon_id != pixel_weapon_id:
            return current_weapon, False

        pixel_display_name = _as_text(pixel_weapon.get("displayName"))
        if pixel_display_name and not _as_text(merged_weapon.get("displayName")):
            merged_weapon["displayName"] = pixel_display_name
            changed = True

        for key in ("level", "levelCap", "baseStatKey", "baseStatValue", "advancedStatKey", "advancedStatValue"):
            if merged_weapon.get(key) not in (None, "", [], {}):
                continue
            pixel_value = pixel_weapon.get(key)
            if pixel_value in (None, "", [], {}):
                continue
            merged_weapon[key] = pixel_value
            changed = True

        if changed:
            merged_weapon["weaponPresent"] = True
            merged_weapon["agentId"] = agent_id
        return merged_weapon, changed

    merged_agents: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    used = False

    for agent in agents:
        payload = dict(agent)
        agent_id = _as_text(payload.get("agentId"))
        seen_agents.add(agent_id)
        pixel_weapon = pixel_weapons_by_agent.get(agent_id)
        if not pixel_weapon:
            merged_agents.append(payload)
            continue

        confidence_by_field = dict(payload.get("confidenceByField") or {})
        field_sources = dict(payload.get("fieldSources") or {})
        existing_weapon = payload.get("weapon") if isinstance(payload.get("weapon"), dict) else {}
        existing_weapon_id = _as_text(existing_weapon.get("weaponId")) if isinstance(existing_weapon, dict) else ""
        previous_weapon_source = _as_text(field_sources.get("weapon"))
        merged_weapon, changed = _merge_weapon_fields(
            current_weapon=existing_weapon if isinstance(existing_weapon, dict) else {},
            pixel_weapon=pixel_weapon,
            agent_id=agent_id,
            allow_identity_fill=not existing_weapon_id or previous_weapon_source in {"", "missing"},
        )
        if not changed:
            merged_agents.append(payload)
            continue

        payload["weapon"] = merged_weapon
        payload["weaponPresent"] = True
        confidence_by_field["weapon"] = round(
            max(float(confidence_by_field.get("weapon", 0.0)), float(pixel_weapon.get("_confidence", 0.0))),
            4,
        )
        field_sources["weapon"] = _weapon_source(pixel_weapon)
        field_sources["weaponPresent"] = "derived_from_amplifier_detail_ocr"
        occupancy = derive_equipment_occupancy(
            weapon=payload.get("weapon") if isinstance(payload.get("weapon"), dict) else None,
            discs=payload.get("discs") if isinstance(payload.get("discs"), list) else None,
            explicit_weapon_present=payload.get("weaponPresent"),
            explicit_slot_occupancy=payload.get("discSlotOccupancy"),
        )
        payload["weaponPresent"] = occupancy["weaponPresent"]
        payload["discSlotOccupancy"] = occupancy["discSlotOccupancy"]
        payload["confidenceByField"] = confidence_by_field
        payload["fieldSources"] = field_sources
        merged_agents.append(payload)
        used = True

    for agent_id, pixel_weapon in sorted(pixel_weapons_by_agent.items()):
        if agent_id in seen_agents:
            continue
        agent_source = _as_text(pixel_weapon.get("agentSource")) or "screen_capture_agent_id"
        agent_confidence = float(pixel_weapon.get("agentConfidence", 0.0) or 0.0)
        payload = _default_agent_payload(agent_id, max(agent_confidence, 0.96))
        payload["fieldSources"]["agentId"] = agent_source
        payload["confidenceByField"]["agentId"] = round(
            max(float(payload["confidenceByField"]["agentId"]), agent_confidence or 0.96),
            4,
        )
        payload["weapon"] = {
            key: value
            for key, value in {
                "weaponId": pixel_weapon.get("weaponId"),
                "displayName": pixel_weapon.get("displayName"),
                "level": pixel_weapon.get("level"),
                "levelCap": pixel_weapon.get("levelCap"),
                "baseStatKey": pixel_weapon.get("baseStatKey"),
                "baseStatValue": pixel_weapon.get("baseStatValue"),
                "advancedStatKey": pixel_weapon.get("advancedStatKey"),
                "advancedStatValue": pixel_weapon.get("advancedStatValue"),
                "weaponPresent": True,
                "agentId": agent_id,
            }.items()
            if value not in (None, "", [], {})
        }
        payload["weaponPresent"] = True
        payload["confidenceByField"]["weapon"] = round(float(pixel_weapon.get("_confidence", 0.0)), 4)
        payload["fieldSources"]["weapon"] = _weapon_source(pixel_weapon)
        payload["fieldSources"]["weaponPresent"] = "derived_from_amplifier_detail_ocr"
        occupancy = derive_equipment_occupancy(
            weapon=payload.get("weapon") if isinstance(payload.get("weapon"), dict) else None,
            discs=payload.get("discs") if isinstance(payload.get("discs"), list) else None,
            explicit_weapon_present=payload.get("weaponPresent"),
            explicit_slot_occupancy=payload.get("discSlotOccupancy"),
        )
        payload["weaponPresent"] = occupancy["weaponPresent"]
        payload["discSlotOccupancy"] = occupancy["discSlotOccupancy"]
        merged_agents.append(payload)
        used = True

    return merged_agents, used


def _enrich_agents_with_agent_detail_pixels(
    agents: list[dict[str, Any]],
    pixel_agent_details: dict[str, dict[str, Any]],
    pixel_agent_details_by_slot: dict[tuple[int | None, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if not pixel_agent_details and not pixel_agent_details_by_slot:
        return agents, False

    merged_agents: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    used = False

    for slot_index, agent in enumerate(agents, start=1):
        payload = dict(agent)
        visible_slot_key = _agent_visible_slot_key(payload, slot_index)
        agent_id = _as_text(payload.get("agentId"))
        detail = (
            pixel_agent_details_by_slot.get(visible_slot_key)
            if visible_slot_key is not None
            else None
        ) or pixel_agent_details.get(agent_id)
        if not detail:
            if agent_id:
                seen_agents.add(agent_id)
            merged_agents.append(payload)
            continue
        _copy_visible_slot_metadata(payload, detail)

        confidence_by_field = dict(payload.get("confidenceByField") or {})
        field_sources = dict(payload.get("fieldSources") or {})
        detail_agent_id = _as_text(detail.get("agentId"))
        if detail_agent_id and detail_agent_id != agent_id:
            payload["agentId"] = detail_agent_id
            confidence_by_field["agentId"] = round(
                max(float(confidence_by_field.get("agentId", 0.0)), float(detail.get("agentConfidence", 0.0) or 0.0)),
                4,
            )
            field_sources["agentId"] = _as_text(detail.get("agentSource")) or "agent_detail_portrait_agreement_onnx_agent_icon"
            agent_id = detail_agent_id
            used = True
        if agent_id:
            seen_agents.add(agent_id)

        if (
            payload.get("level") is None
            and detail.get("level") is not None
            and detail.get("levelCap") is not None
            and float(detail.get("levelConfidence") or 0.0) >= _AGENT_DETAIL_LEVEL_CONFIDENCE_THRESHOLD
        ):
            payload["level"] = detail.get("level")
            payload["levelCap"] = detail.get("levelCap")
            confidence_by_field["level"] = round(float(detail.get("levelConfidence") or 0.0), 4)
            field_sources["level"] = "agent_detail_digit_ocr"
            field_sources["levelCap"] = "agent_detail_digit_ocr"
            used = True

        if (
            payload.get("mindscape") is None
            and detail.get("mindscape") is not None
            and detail.get("mindscapeCap") is not None
            and float(detail.get("mindscapeConfidence") or 0.0) >= 0.50
        ):
            payload["mindscape"] = detail.get("mindscape")
            payload["mindscapeCap"] = detail.get("mindscapeCap")
            confidence_by_field["mindscape"] = round(float(detail.get("mindscapeConfidence") or 0.0), 4)
            field_sources["mindscape"] = "agent_detail_digit_ocr"
            field_sources["mindscapeCap"] = "agent_detail_digit_ocr"
            used = True

        if (
            not isinstance(payload.get("stats"), dict) or not payload.get("stats")
        ) and isinstance(detail.get("stats"), dict) and detail.get("stats"):
            payload["stats"] = dict(detail.get("stats") or {})
            confidence_by_field["stats"] = round(float(detail.get("statsConfidence") or 0.0), 4)
            field_sources["stats"] = "agent_detail_digit_templates"
            used = True

        payload["confidenceByField"] = confidence_by_field
        payload["fieldSources"] = field_sources
        merged_agents.append(payload)

    for agent_id, detail in sorted(pixel_agent_details.items()):
        if agent_id in seen_agents:
            continue
        agent_source = _as_text(detail.get("agentSource")) or "screen_capture_agent_id"
        agent_confidence = float(detail.get("agentConfidence", 0.0) or 0.0)
        payload = _default_agent_payload(agent_id, max(agent_confidence, 0.96))
        payload["fieldSources"]["agentId"] = agent_source
        payload["confidenceByField"]["agentId"] = round(
            max(float(payload["confidenceByField"]["agentId"]), agent_confidence or 0.96),
            4,
        )
        _copy_visible_slot_metadata(payload, detail)
        if (
            detail.get("level") is not None
            and detail.get("levelCap") is not None
            and float(detail.get("levelConfidence") or 0.0) >= _AGENT_DETAIL_LEVEL_CONFIDENCE_THRESHOLD
        ):
            payload["level"] = detail.get("level")
            payload["levelCap"] = detail.get("levelCap")
            payload["confidenceByField"]["level"] = round(float(detail.get("levelConfidence") or 0.0), 4)
            payload["fieldSources"]["level"] = "agent_detail_digit_ocr"
            payload["fieldSources"]["levelCap"] = "agent_detail_digit_ocr"
        if (
            detail.get("mindscape") is not None
            and detail.get("mindscapeCap") is not None
            and float(detail.get("mindscapeConfidence") or 0.0) >= 0.50
        ):
            payload["mindscape"] = detail.get("mindscape")
            payload["mindscapeCap"] = detail.get("mindscapeCap")
            payload["confidenceByField"]["mindscape"] = round(float(detail.get("mindscapeConfidence") or 0.0), 4)
            payload["fieldSources"]["mindscape"] = "agent_detail_digit_ocr"
            payload["fieldSources"]["mindscapeCap"] = "agent_detail_digit_ocr"
        if isinstance(detail.get("stats"), dict) and detail.get("stats"):
            payload["stats"] = dict(detail.get("stats") or {})
            payload["confidenceByField"]["stats"] = round(float(detail.get("statsConfidence") or 0.0), 4)
            payload["fieldSources"]["stats"] = "agent_detail_digit_templates"
        merged_agents.append(payload)
        used = True

    return merged_agents, used


def _extract_agents(
    payload: Iterable[Dict[str, Any]],
    session_context: Dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    low_conf_reasons: list[str] = []
    occupancy_by_agent_raw = session_context.get("equipmentOccupancyByAgent")
    occupancy_by_agent = (
        occupancy_by_agent_raw
        if isinstance(occupancy_by_agent_raw, dict)
        else {}
    )

    for index, raw in enumerate(payload):
        agent_id = str(raw.get("agentId", "")).strip()
        if not agent_id:
            low_conf_reasons.append(f"agent[{index}].agentId_missing")
            continue

        level_raw = raw.get("level")
        level = float(level_raw) if isinstance(level_raw, (int, float)) else None
        level_cap_raw = raw.get("levelCap")
        level_cap = float(level_cap_raw) if isinstance(level_cap_raw, (int, float)) else None
        level_conf = 0.24 if level is None or level_cap is None else (0.88 + 0.12 * _normalized_fraction(level, 1.0, 60.0))
        if level is None or level_cap is None:
            low_conf_reasons.append(f"{agent_id}.level_missing")

        mindscape_raw = raw.get("mindscape")
        mindscape = float(mindscape_raw) if isinstance(mindscape_raw, (int, float)) else None
        mindscape_cap_raw = raw.get("mindscapeCap")
        mindscape_cap = float(mindscape_cap_raw) if isinstance(mindscape_cap_raw, (int, float)) else None
        mindscape_conf = (
            0.24 if mindscape is None or mindscape_cap is None else (0.88 + 0.1 * _normalized_fraction(mindscape, 0.0, 6.0))
        )
        if mindscape is None or mindscape_cap is None:
            low_conf_reasons.append(f"{agent_id}.mindscape_missing")

        stats = raw.get("stats")
        if not isinstance(stats, dict):
            stats = {}

        weapon = raw.get("weapon")
        discs = raw.get("discs")
        has_weapon = _has_meaningful_weapon_payload(weapon)
        has_discs = _has_meaningful_disc_payload(discs)
        if not has_weapon:
            low_conf_reasons.append(f"{agent_id}.weapon_missing")
            weapon = {}
        if not has_discs:
            low_conf_reasons.append(f"{agent_id}.discs_missing")
            discs = []

        occupancy_override = occupancy_by_agent.get(agent_id)
        if not isinstance(occupancy_override, dict):
            occupancy_override = {}
        raw_has_weapon_present = "weaponPresent" in raw
        raw_has_slot_occupancy = "discSlotOccupancy" in raw
        override_has_weapon_present = "weaponPresent" in occupancy_override
        override_has_slot_occupancy = "discSlotOccupancy" in occupancy_override
        occupancy = derive_equipment_occupancy(
            weapon=weapon,
            discs=discs,
            explicit_weapon_present=raw.get("weaponPresent", occupancy_override.get("weaponPresent")),
            explicit_slot_occupancy=raw.get("discSlotOccupancy", occupancy_override.get("discSlotOccupancy")),
        )
        occupancy_conf = 0.92 if occupancy_override else (0.76 if has_weapon or has_discs else 0.18)
        known_empty_weapon = (
            not has_weapon
            and (raw_has_weapon_present or override_has_weapon_present)
            and occupancy.get("weaponPresent") is False
        )
        known_empty_discs = (
            not has_discs
            and (raw_has_slot_occupancy or override_has_slot_occupancy)
            and _all_disc_slots_empty(occupancy.get("discSlotOccupancy"))
        )
        weapon_conf = _known_empty_equipment_confidence(occupancy_conf) if known_empty_weapon else (0.9 if has_weapon else 0.22)
        disc_conf = _known_empty_equipment_confidence(occupancy_conf) if known_empty_discs else (0.9 if has_discs else 0.22)

        confidence_by_field = {
            "agentId": float(raw.get("agentConfidence", 0.99)),
            "level": round(level_conf, 4),
            "mindscape": round(mindscape_conf, 4),
            "weapon": round(weapon_conf, 4),
            "discs": round(disc_conf, 4),
            "occupancy": round(occupancy_conf, 4),
        }
        field_sources = {
            "agentId": "session_payload",
            "level": "session_payload" if level is not None else "missing",
            "levelCap": "session_payload" if level_cap is not None else "missing",
            "mindscape": "session_payload" if mindscape is not None else "missing",
            "mindscapeCap": "session_payload" if mindscape_cap is not None else "missing",
            "stats": "session_payload" if stats else "missing",
            "weapon": (
                "known_empty_from_equipment_occupancy"
                if known_empty_weapon
                else ("session_payload" if weapon else "missing")
            ),
            "discs": (
                "known_empty_from_equipment_occupancy"
                if known_empty_discs
                else ("session_payload" if discs else "missing")
            ),
            "weaponPresent": (
                "equipment_occupancy_payload"
                if raw_has_weapon_present or override_has_weapon_present
                else ("derived_from_weapon_payload" if weapon else "missing")
            ),
            "discSlotOccupancy": (
                "equipment_occupancy_payload"
                if raw_has_slot_occupancy or override_has_slot_occupancy
                else ("derived_from_discs_payload" if discs else "missing")
            ),
        }

        parsed_agent = {
            "agentId": agent_id,
            "level": level,
            "levelCap": level_cap,
            "mindscape": mindscape,
            "mindscapeCap": mindscape_cap,
            "stats": stats,
            "weapon": weapon,
            "discs": discs,
            "weaponPresent": occupancy["weaponPresent"],
            "discSlotOccupancy": occupancy["discSlotOccupancy"],
            "confidenceByField": confidence_by_field,
            "fieldSources": field_sources,
        }
        if isinstance(raw, dict):
            _copy_visible_slot_metadata(parsed_agent, raw)
        results.append(parsed_agent)

    if not results:
        low_conf_reasons.append("agents_empty_after_parse")

    return results, low_conf_reasons


def _top_level_equipment_source(agents: list[dict[str, Any]]) -> str:
    has_payload_equipment = False
    has_payload_occupancy = False
    has_derived_occupancy = False
    has_pixel_occupancy = False
    has_onnx_disk_detail = False
    has_merged_disk_detail = False
    has_amplifier_detail_ocr = False
    for agent in agents:
        field_sources = agent.get("fieldSources")
        if not isinstance(field_sources, dict):
            continue
        if field_sources.get("discs") == "onnx_disk_detail":
            has_onnx_disk_detail = True
        if field_sources.get("discs") == "merged_session_payload_and_onnx_disk_detail":
            has_merged_disk_detail = True
        if field_sources.get("weapon") in {"amplifier_detail_title_ocr", "amplifier_detail_ocr"}:
            has_amplifier_detail_ocr = True
        if field_sources.get("weapon") == "session_payload" or field_sources.get("discs") == "session_payload":
            has_payload_equipment = True
        if (
            field_sources.get("weaponPresent") == "equipment_occupancy_payload"
            or field_sources.get("discSlotOccupancy") == "equipment_occupancy_payload"
        ):
            has_payload_occupancy = True
        if (
            field_sources.get("weaponPresent") == "pixel_equipment_overview"
            or field_sources.get("discSlotOccupancy") == "pixel_equipment_overview"
        ):
            has_pixel_occupancy = True
        if (
            field_sources.get("weaponPresent") == "derived_from_weapon_payload"
            or field_sources.get("discSlotOccupancy") == "derived_from_discs_payload"
        ):
            has_derived_occupancy = True

    if has_merged_disk_detail:
        return "merged_session_payload_and_onnx_disk_detail"
    if has_onnx_disk_detail and has_amplifier_detail_ocr:
        return "pixel_equipment_detail_ocr"
    if has_onnx_disk_detail:
        return "onnx_disk_detail"
    if has_amplifier_detail_ocr:
        return "amplifier_detail_ocr"
    if has_pixel_occupancy:
        return "pixel_equipment_overview"
    if has_payload_equipment:
        return "session_payload"
    if has_payload_occupancy:
        return "equipment_occupancy_payload"
    if has_derived_occupancy:
        return "derived_from_payload"
    return "missing"


def _filter_resolved_low_conf_reasons(
    agents: list[dict[str, Any]],
    reasons: list[str],
) -> list[str]:
    if not reasons:
        return []
    by_agent: dict[str, dict[str, Any]] = {}
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        agent_id = _as_text(agent.get("agentId"))
        if agent_id:
            by_agent[agent_id] = agent

    filtered: list[str] = []
    for reason in reasons:
        if not isinstance(reason, str):
            continue
        matched = False
        if reason == "agents_empty_after_parse" and by_agent:
            matched = True
        if reason.startswith("disk_detail_low_conf:"):
            parts = reason.split(":")
            agent = by_agent.get(parts[1] if len(parts) > 1 else "")
            confidence_by_field = agent.get("confidenceByField") if isinstance(agent, dict) else None
            disc_confidence = (
                float(confidence_by_field.get("discs", 0.0) or 0.0)
                if isinstance(confidence_by_field, dict)
                else 0.0
            )
            if agent is not None and disc_confidence >= 0.80 and _disc_consensus_confidence(agent.get("discs")) >= 0.80:
                matched = True
        if not matched and ":equipment_overview_weapon_presence_ambiguous" in reason:
            agent_id = reason.split(":", 1)[0]
            agent = by_agent.get(agent_id)
            if isinstance(agent, dict):
                weapon = agent.get("weapon")
                if (
                    (isinstance(weapon, dict) and _as_text(weapon.get("weaponId")) != "")
                    or agent.get("weaponPresent") is False
                ):
                    matched = True
        if not matched and ":equipment_overview_slot_ambiguous:" in reason:
            agent_id, slot_text = reason.split(":equipment_overview_slot_ambiguous:", 1)
            slot_index = _as_slot_index(slot_text)
            agent = by_agent.get(agent_id)
            if slot_index is not None and isinstance(agent, dict):
                discs = agent.get("discs")
                slot_occupancy = agent.get("discSlotOccupancy")
                if (
                    isinstance(discs, list)
                    and any(
                        isinstance(disc, dict)
                        and _as_slot_index(disc.get("slot")) == slot_index
                        and _as_text(disc.get("setId")) != ""
                        for disc in discs
                    )
                ):
                    matched = True
                elif isinstance(slot_occupancy, dict) and str(slot_index) in slot_occupancy:
                    matched = True
        if not matched and reason.startswith("amplifier_detail_unclassified:"):
            agent = by_agent.get(reason.split(":", 1)[1])
            if isinstance(agent, dict) and agent.get("weaponPresent") is False:
                matched = True
        for suffix, predicate in (
            (".weapon_missing", lambda agent: isinstance(agent.get("weapon"), dict) and _as_text(agent.get("weapon", {}).get("weaponId")) != ""),
            (".discs_missing", lambda agent: isinstance(agent.get("discs"), list) and len(agent.get("discs") or []) > 0),
            (".level_missing", lambda agent: agent.get("level") is not None and agent.get("levelCap") is not None),
            (".mindscape_missing", lambda agent: agent.get("mindscape") is not None and agent.get("mindscapeCap") is not None),
        ):
            if not reason.endswith(suffix):
                continue
            agent_id = reason[: -len(suffix)]
            agent = by_agent.get(agent_id)
            if agent is not None and predicate(agent):
                matched = True
            break
        if not matched and reason.endswith(".weapon_missing"):
            agent = by_agent.get(reason[: -len(".weapon_missing")])
            if agent is not None and agent.get("weaponPresent") is False:
                matched = True
        if not matched and reason.endswith(".discs_missing"):
            agent = by_agent.get(reason[: -len(".discs_missing")])
            slot_occupancy = agent.get("discSlotOccupancy") if isinstance(agent, dict) else None
            if isinstance(slot_occupancy, dict) and not any(bool(value) for value in slot_occupancy.values()):
                matched = True
        if not matched:
            filtered.append(reason)
    return filtered


def _drop_stale_top_level_confidence_reasons(
    reasons: list[str],
    top_confidence: dict[str, float],
) -> list[str]:
    if not reasons:
        return []

    thresholds = {
        "uid_low_confidence": ("uid", 0.9),
        "agents_low_confidence": ("agents", 0.9),
        "equipment_low_confidence": ("equipment", 0.85),
    }
    filtered: list[str] = []
    for reason in reasons:
        metric = thresholds.get(reason)
        if metric is None:
            filtered.append(reason)
            continue
        field_name, threshold = metric
        if float(top_confidence.get(field_name, 0.0) or 0.0) < threshold:
            filtered.append(reason)
    return filtered


def _infer_full_roster_coverage(session_context: Dict[str, Any]) -> bool:
    for key in ("fullRosterCoverage", "fullRosterTerminalSliceReached", "terminalSliceReached"):
        value = session_context.get(key)
        if isinstance(value, bool):
            return value

    raw_captures = session_context.get("screenCaptures")
    if not isinstance(raw_captures, list):
        return False

    has_roster_capture = any(
        isinstance(capture, dict) and _as_text(capture.get("role")) == "roster"
        for capture in raw_captures
    )
    if not has_roster_capture:
        return False

    icon_payload = session_context.get("agentIconPaths")
    if not isinstance(icon_payload, list) or not icon_payload:
        return False

    page_counts: dict[int, int] = {}
    observed_page_capacity = 0
    for entry in icon_payload:
        if not isinstance(entry, dict):
            continue
        page_index = _as_slot_index(entry.get("pageIndex"))
        if page_index is None or page_index < 0:
            continue
        page_counts[page_index] = page_counts.get(page_index, 0) + 1
        observed_page_capacity = max(
            observed_page_capacity,
            _as_slot_index(entry.get("rosterPageSlotIndex")) or 0,
        )

    if not page_counts:
        return False

    if observed_page_capacity <= 0:
        observed_page_capacity = max(page_counts.values())
    if observed_page_capacity <= 0:
        return False

    last_page = max(page_counts)
    last_page_count = page_counts[last_page]
    return 0 < last_page_count < observed_page_capacity


def scan_roster(
    session_context: Dict[str, Any],
    calibration: Dict[str, Any] | None,
    locale: str,
    resolution: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    metadata = get_model_metadata(DEFAULT_MODEL_VERSION)

    if not bool(session_context.get("inputLockActive", False)):
        raise ScanFailure(
            ScanFailureCode.INPUT_LOCK_NOT_ACTIVE,
            "Scan requires active host input lock.",
            ["input_lock_not_active"],
        )

    normalized_locale = _normalize_locale(locale)
    normalized_resolution = normalize_runtime_resolution(session_context, resolution)
    if not normalized_locale or not normalized_resolution:
        raise ScanFailure(
            ScanFailureCode.UNSUPPORTED_LAYOUT,
            "Unsupported locale or layout family.",
            [
                f"locale={locale}",
                f"resolution={resolution}",
                "supported_locales=RU|EN",
                "layout_family=16:9_fractional_crops",
            ],
        )

    session_context = normalize_runtime_captures(session_context, normalized_resolution)

    anchors = session_context.get("anchors") if isinstance(session_context.get("anchors"), dict) else {}
    required_anchors = (
        calibration.get("requiredAnchors", ["profile", "agents", "equipment"])
        if isinstance(calibration, dict)
        else ["profile", "agents", "equipment"]
    )
    missing = [anchor for anchor in required_anchors if not bool(anchors.get(anchor, False))]
    if missing:
        raise ScanFailure(
            ScanFailureCode.ANCHOR_MISMATCH,
            "Anchor validation failed.",
            [f"anchor_missing:{value}" for value in missing],
        )

    low_conf_reasons: list[str] = []
    uid_from_image, uid_conf_from_image, uid_source_from_image, uid_reasons = _select_uid_from_image(session_context)
    low_conf_reasons.extend(uid_reasons)
    if uid_from_image and uid_conf_from_image is not None:
        uid = uid_from_image
        uid_confidence = uid_conf_from_image
    else:
        uid = ""
        uid_confidence = 0.0

    if not uid:
        low_conf_reasons.append("uid_missing")
    uid_source = uid_source_from_image if uid_from_image else "missing"

    icon_agents, icon_reasons = _agents_from_icons(session_context)
    low_conf_reasons.extend(icon_reasons)

    raw_agents = session_context.get("agents")
    if not isinstance(raw_agents, list):
        raw_agents = []
    parsed_agents, parsed_reasons = _extract_agents(raw_agents, session_context)
    low_conf_reasons.extend(parsed_reasons)

    if parsed_agents and icon_agents and len(parsed_agents) != len(icon_agents):
        low_conf_reasons.append("agent_count_mismatch_between_icon_and_payload")

    agents, merge_reasons = _merge_agents(parsed_agents, icon_agents)
    low_conf_reasons.extend(merge_reasons)
    agent_ids_by_slot = _agent_ids_by_slot(agents)
    pixel_agent_details, pixel_agent_details_by_slot, pixel_agent_detail_reasons = _pixel_agent_details_from_captures(
        session_context,
        agent_ids_by_slot,
    )
    low_conf_reasons.extend(pixel_agent_detail_reasons)
    agent_ids_by_slot = _merge_agent_ids_by_slot_from_details(agent_ids_by_slot, pixel_agent_details_by_slot)
    pixel_equipment_occupancy_by_agent, pixel_equipment_occupancy_reasons = _pixel_equipment_occupancy_from_captures(
        session_context,
        agent_ids_by_slot,
    )
    low_conf_reasons.extend(pixel_equipment_occupancy_reasons)
    pixel_weapons_by_agent, pixel_weapon_reasons = _pixel_weapons_from_captures(
        session_context,
        agent_ids_by_slot,
        pixel_equipment_occupancy_by_agent,
        normalized_locale,
    )
    low_conf_reasons.extend(pixel_weapon_reasons)
    pixel_discs_by_agent, pixel_disc_reasons = _pixel_discs_from_captures(
        session_context,
        agent_ids_by_slot,
        normalized_locale,
    )
    low_conf_reasons.extend(pixel_disc_reasons)
    agents, used_pixel_agent_details = _enrich_agents_with_agent_detail_pixels(
        agents,
        pixel_agent_details,
        pixel_agent_details_by_slot,
    )
    agents, used_pixel_occupancy = _enrich_agents_with_pixel_equipment_occupancy(
        agents,
        pixel_equipment_occupancy_by_agent,
    )
    agents, used_pixel_weapons = _enrich_agents_with_pixel_weapons(agents, pixel_weapons_by_agent)
    agents, used_pixel_discs = _enrich_agents_with_pixel_discs(agents, pixel_discs_by_agent)
    if not agents:
        low_conf_reasons.append("agents_missing")
    else:
        low_conf_reasons = _filter_resolved_low_conf_reasons(agents, low_conf_reasons)

    region = _normalize_region(
        str(session_context.get("region") or session_context.get("regionHint") or "OTHER")
    )

    top_confidence = {
        "uid": round(uid_confidence, 4),
        "region": 0.97 if region != "OTHER" else 0.83,
        "agents": round(
            sum(agent["confidenceByField"]["agentId"] for agent in agents) / max(len(agents), 1), 4
        ),
        "equipment": round(
            sum(
                (agent["confidenceByField"]["weapon"] + agent["confidenceByField"]["discs"]) / 2.0
                for agent in agents
            )
            / max(len(agents), 1),
            4,
        ),
    }
    low_conf_reasons = _drop_stale_top_level_confidence_reasons(low_conf_reasons, top_confidence)
    screen_captures = session_context.get("screenCaptures")
    has_roster_capture = isinstance(screen_captures, list) and any(
        isinstance(capture, dict) and str(capture.get("role", "")).strip() == "roster"
        for capture in screen_captures
    )
    has_direct_crops = (
        bool(session_context.get("uidImagePath"))
        or bool(session_context.get("agentIconPaths"))
        or bool(pixel_agent_details)
        or bool(pixel_weapons_by_agent)
        or bool(pixel_discs_by_agent)
    )
    field_sources = {
        "uid": uid_source,
        "region": (
            "session_region"
            if session_context.get("region")
            else ("session_region_hint" if session_context.get("regionHint") else "default_other")
        ),
        "agents": (
            "merged_pixels_and_payload"
            if parsed_agents and (icon_agents or used_pixel_agent_details or used_pixel_weapons or used_pixel_discs)
            else (
                "onnx_agent_icon"
                if icon_agents
                else (
                    "agent_detail_portrait_agreement_onnx_agent_icon"
                    if used_pixel_agent_details
                    else ("session_payload" if parsed_agents else ("pixel_equipment_overview" if used_pixel_occupancy else "missing"))
                )
            )
        ),
        "equipment": _top_level_equipment_source(agents),
        "capture": (
            "runtime_roster_screen"
            if has_roster_capture
            else ("provided_crops" if has_direct_crops else "missing")
        ),
    }
    capabilities = {
        "uidFromPixels": uid_source == "onnx_uid_digits",
        "agentIdsFromPixels": bool(icon_agents or used_pixel_agent_details or used_pixel_occupancy or used_pixel_weapons or used_pixel_discs),
        "equipmentFromPixels": bool(used_pixel_occupancy or used_pixel_weapons or used_pixel_discs),
        "equipmentOverviewFromPixels": bool(used_pixel_occupancy),
        "agentDetailsFromPixels": bool(used_pixel_agent_details),
        "fullRosterCoverage": _infer_full_roster_coverage(session_context),
        "rawRosterCaptureAvailable": has_roster_capture,
    }

    if top_confidence["uid"] < 0.9:
        low_conf_reasons.append("uid_low_confidence")
    if top_confidence["agents"] < 0.9:
        low_conf_reasons.append("agents_low_confidence")
    if top_confidence["equipment"] < 0.85:
        low_conf_reasons.append("equipment_low_confidence")

    timing_ms = round((time.perf_counter() - started) * 1000.0, 2)

    result = {
        "uid": uid,
        "region": region,
        "modelVersion": metadata["modelVersion"],
        "dataVersion": metadata["dataVersion"],
        "scanMeta": "hybrid_onnx_plus_rules",
        "confidenceByField": top_confidence,
        "fieldSources": field_sources,
        "capabilities": capabilities,
        "agents": agents,
        "lowConfReasons": sorted(set(low_conf_reasons)),
        "timingMs": timing_ms,
        "resolution": normalized_resolution,
        "locale": normalized_locale,
    }

    if low_conf_reasons:
        raise ScanFailure(
            ScanFailureCode.LOW_CONFIDENCE,
            "Scan finished with low confidence.",
            sorted(set(low_conf_reasons)),
            partial_result=result,
        )

    return result


def run_scan(seed: str, region: str, full_sync: bool) -> Dict[str, Any]:
    rng = random.Random(seed)
    agents = [
        {
            "agentId": "agent_anby",
            "level": rng.randint(40, 60),
            "levelCap": 60,
            "mindscape": rng.randint(0, 6),
            "mindscapeCap": 6,
            "weapon": {"weaponId": "amp_starlight_engine", "level": rng.randint(40, 60)},
            "discs": [
                {"slot": 1, "setId": "set_woodpecker", "level": rng.randint(12, 15)},
                {"slot": 2, "setId": "set_woodpecker", "level": rng.randint(12, 15)},
            ],
        },
        {
            "agentId": "agent_nicole",
            "level": rng.randint(40, 60),
            "levelCap": 60,
            "mindscape": rng.randint(0, 6),
            "mindscapeCap": 6,
            "weapon": {"weaponId": "amp_rainforest", "level": rng.randint(40, 60)},
            "discs": [
                {"slot": 1, "setId": "set_swing_jazz", "level": rng.randint(12, 15)},
                {"slot": 2, "setId": "set_swing_jazz", "level": rng.randint(12, 15)},
            ],
        },
        {
            "agentId": "agent_ellen",
            "level": rng.randint(40, 60),
            "levelCap": 60,
            "mindscape": rng.randint(0, 6),
            "mindscapeCap": 6,
            "weapon": {"weaponId": "amp_deep_sea", "level": rng.randint(40, 60)},
            "discs": [
                {"slot": 1, "setId": "set_polar_metal", "level": rng.randint(12, 15)},
                {"slot": 2, "setId": "set_polar_metal", "level": rng.randint(12, 15)},
            ],
        },
    ]

    uid_sample = str(rng.randint(100000000, 999999999))
    metadata = get_model_metadata(DEFAULT_MODEL_VERSION)
    visible_agents = agents if full_sync else agents[:2]
    average_level_confidence = round(
        sum(0.93 for _ in visible_agents) / max(len(visible_agents), 1),
        4,
    )
    average_equipment_confidence = round(
        sum(0.91 for _ in visible_agents) / max(len(visible_agents), 1),
        4,
    )
    return {
        "uid": uid_sample,
        "region": _normalize_region(region),
        "modelVersion": metadata["modelVersion"],
        "dataVersion": metadata["dataVersion"],
        "scanMeta": "demo_seeded_payload",
        "confidenceByField": {
            "uid": 0.995,
            "region": 0.97 if _normalize_region(region) != "OTHER" else 0.83,
            "agents": average_level_confidence,
            "equipment": average_equipment_confidence,
        },
        "fieldSources": {
            "uid": "demo_seed",
            "region": "demo_seed",
            "agents": "demo_seed",
            "equipment": "demo_seed",
            "capture": "demo_seed",
        },
        "capabilities": {
            "uidFromPixels": False,
            "agentIdsFromPixels": False,
            "equipmentFromPixels": False,
            "agentDetailsFromPixels": False,
            "fullRosterCoverage": bool(full_sync),
            "rawRosterCaptureAvailable": False,
        },
        "agents": visible_agents,
        "lowConfReasons": [],
        "timingMs": 0.0,
        "resolution": "1080p",
        "locale": "EN",
    }
