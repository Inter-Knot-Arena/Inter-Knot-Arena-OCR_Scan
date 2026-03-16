from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from .model_runtime import (
    ModelRegistry,
    classify_agent_icon,
    classify_disk_detail,
    classify_uid_digits,
    get_model_metadata,
    segment_uid_digits,
)
from .screen_runtime import derive_equipment_occupancy, normalize_runtime_captures, normalize_runtime_resolution

SUPPORTED_LOCALES = {"RU", "EN"}
SUPPORTED_REGIONS = {"NA", "EU", "ASIA", "SEA", "OTHER"}
DEFAULT_MODEL_VERSION = "ocr-hybrid-v1.2"


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


def _select_uid_from_image(session_context: Dict[str, Any]) -> tuple[str | None, float | None, list[str]]:
    reasons: list[str] = []
    uid_path_raw = session_context.get("uidImagePath")
    if not isinstance(uid_path_raw, str) or not uid_path_raw.strip():
        return None, None, reasons
    uid_path = Path(uid_path_raw)
    if not uid_path.exists():
        reasons.append("uid_image_not_found")
        return None, None, reasons

    if not ModelRegistry.has_uid_model():
        reasons.append("uid_model_missing")
        return None, None, reasons

    image = cv2.imread(str(uid_path), cv2.IMREAD_COLOR)
    if image is None:
        reasons.append("uid_image_decode_failed")
        return None, None, reasons

    digit_images = segment_uid_digits(image)
    if len(digit_images) < 6:
        reasons.append("uid_digit_segmentation_failed")
        return None, None, reasons

    uid, confidence = classify_uid_digits(digit_images)
    digits_only = _digits_only(uid)
    if not (6 <= len(digits_only) <= 12):
        reasons.append("uid_digit_length_invalid")
        return None, None, reasons
    return digits_only, confidence, reasons


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
            "level": 0.62,
            "mindscape": 0.62,
            "weapon": 0.58,
            "discs": 0.58,
            "occupancy": 0.52,
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
        agents.append(_default_agent_payload(prediction.label, prediction.confidence))
        if prediction.confidence < 0.72:
            reasons.append(f"agent_icon_low_conf:{prediction.label}")

    return agents, reasons


def _pixel_discs_from_captures(session_context: Dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
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
    for index, entry in enumerate(disk_entries):
        path_value = _as_text(entry.get("path"))
        agent_id = _as_text(entry.get("agentId") or entry.get("focusAgentId"))
        slot_index = _as_slot_index(entry.get("slotIndex"))
        if not path_value:
            reasons.append(f"disk_detail_missing_path:{index}")
            continue
        if not agent_id:
            reasons.append(f"disk_detail_missing_agent:{index}")
            continue
        if slot_index is None or slot_index < 1 or slot_index > 6:
            reasons.append(f"disk_detail_missing_slot:{agent_id}:{index}")
            continue

        image = cv2.imread(str(Path(path_value)), cv2.IMREAD_COLOR)
        if image is None:
            reasons.append(f"disk_detail_decode_failed:{agent_id}:{slot_index}")
            continue

        prediction = classify_disk_detail(image)
        if prediction.confidence < 0.72:
            reasons.append(f"disk_detail_low_conf:{agent_id}:{slot_index}:{prediction.label}")

        agent_slots = by_agent_slot.setdefault(agent_id, {})
        candidate = {
            "slot": slot_index,
            "setId": prediction.label,
            "agentId": agent_id,
            "_confidence": float(prediction.confidence),
        }
        existing = agent_slots.get(slot_index)
        if existing is None or float(candidate["_confidence"]) > float(existing["_confidence"]):
            agent_slots[slot_index] = candidate

    output: dict[str, list[dict[str, Any]]] = {}
    for agent_id, slot_map in by_agent_slot.items():
        output[agent_id] = [slot_map[slot] for slot in sorted(slot_map)]
    return output, reasons


def _merge_agents(
    parsed_agents: list[dict[str, Any]],
    icon_agents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not parsed_agents:
        return icon_agents
    if not icon_agents:
        return parsed_agents

    merged: list[dict[str, Any]] = []
    total = max(len(parsed_agents), len(icon_agents))
    for index in range(total):
        parsed = parsed_agents[index] if index < len(parsed_agents) else None
        icon = icon_agents[index] if index < len(icon_agents) else None
        if parsed is None:
            if icon is not None:
                merged.append(icon)
            continue
        if icon is None:
            merged.append(parsed)
            continue

        merged_payload = dict(parsed)
        merged_confidence = dict(parsed.get("confidenceByField") or {})
        merged_field_sources = dict(parsed.get("fieldSources") or {})
        icon_confidence = icon.get("confidenceByField")
        icon_agent_id = str(icon.get("agentId", "")).strip()
        if icon_agent_id:
            merged_payload["agentId"] = icon_agent_id
            merged_field_sources["agentId"] = "onnx_agent_icon"
        if isinstance(icon_confidence, dict):
            agent_conf = icon_confidence.get("agentId")
            if isinstance(agent_conf, (int, float)):
                merged_confidence["agentId"] = round(float(agent_conf), 4)
        merged_payload["confidenceByField"] = merged_confidence
        merged_payload["fieldSources"] = merged_field_sources
        merged.append(merged_payload)

    return merged


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
        previous_disc_source = _as_text(field_sources.get("discs"))
        confidence_by_field["discs"] = round(max(float(confidence_by_field.get("discs", 0.0)), avg_confidence), 4)
        confidence_by_field["occupancy"] = round(max(float(confidence_by_field.get("occupancy", 0.0)), avg_confidence), 4)
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
        payload = _default_agent_payload(agent_id, avg_confidence)
        payload["discs"] = [
            {key: value for key, value in disc.items() if not str(key).startswith("_")}
            for disc in pixel_discs
        ]
        payload["confidenceByField"]["discs"] = round(avg_confidence, 4)
        payload["confidenceByField"]["occupancy"] = round(avg_confidence, 4)
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
        level_conf = 0.84 if level is None else (0.88 + 0.12 * _normalized_fraction(level, 1.0, 60.0))
        if level is None:
            low_conf_reasons.append(f"{agent_id}.level_missing")

        mindscape_raw = raw.get("mindscape")
        mindscape = float(mindscape_raw) if isinstance(mindscape_raw, (int, float)) else None
        mindscape_cap_raw = raw.get("mindscapeCap")
        mindscape_cap = float(mindscape_cap_raw) if isinstance(mindscape_cap_raw, (int, float)) else None
        mindscape_conf = 0.82 if mindscape is None else (0.88 + 0.1 * _normalized_fraction(mindscape, 0.0, 6.0))
        if mindscape is None:
            low_conf_reasons.append(f"{agent_id}.mindscape_missing")

        stats = raw.get("stats")
        if not isinstance(stats, dict):
            stats = {}

        weapon = raw.get("weapon")
        discs = raw.get("discs")
        weapon_conf = 0.9 if isinstance(weapon, dict) else 0.78
        disc_conf = 0.9 if isinstance(discs, list) else 0.76
        if not isinstance(weapon, dict):
            low_conf_reasons.append(f"{agent_id}.weapon_missing")
            weapon = {}
        if not isinstance(discs, list):
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
        occupancy_conf = 0.92 if occupancy_override else 0.84

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
            "weapon": "session_payload" if weapon else "missing",
            "discs": "session_payload" if discs else "missing",
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

        results.append(
            {
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
        )

    if not results:
        low_conf_reasons.append("agents_empty_after_parse")

    return results, low_conf_reasons


def _top_level_equipment_source(agents: list[dict[str, Any]]) -> str:
    has_payload_equipment = False
    has_payload_occupancy = False
    has_derived_occupancy = False
    has_onnx_disk_detail = False
    has_merged_disk_detail = False
    for agent in agents:
        field_sources = agent.get("fieldSources")
        if not isinstance(field_sources, dict):
            continue
        if field_sources.get("discs") == "onnx_disk_detail":
            has_onnx_disk_detail = True
        if field_sources.get("discs") == "merged_session_payload_and_onnx_disk_detail":
            has_merged_disk_detail = True
        if field_sources.get("weapon") == "session_payload" or field_sources.get("discs") == "session_payload":
            has_payload_equipment = True
        if (
            field_sources.get("weaponPresent") == "equipment_occupancy_payload"
            or field_sources.get("discSlotOccupancy") == "equipment_occupancy_payload"
        ):
            has_payload_occupancy = True
        if (
            field_sources.get("weaponPresent") == "derived_from_weapon_payload"
            or field_sources.get("discSlotOccupancy") == "derived_from_discs_payload"
        ):
            has_derived_occupancy = True

    if has_merged_disk_detail:
        return "merged_session_payload_and_onnx_disk_detail"
    if has_onnx_disk_detail:
        return "onnx_disk_detail"
    if has_payload_equipment:
        return "session_payload"
    if has_payload_occupancy:
        return "equipment_occupancy_payload"
    if has_derived_occupancy:
        return "derived_from_payload"
    return "missing"


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
    if not normalized_locale:
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
    uid_from_image, uid_conf_from_image, uid_reasons = _select_uid_from_image(session_context)
    low_conf_reasons.extend(uid_reasons)
    if uid_from_image and uid_conf_from_image is not None:
        uid = uid_from_image
        uid_confidence = uid_conf_from_image
    else:
        uid_candidates = session_context.get("uidCandidates")
        if not isinstance(uid_candidates, list):
            uid_candidates = []
        uid, uid_confidence = _select_uid_from_candidates((str(value) for value in uid_candidates))

    if not uid:
        low_conf_reasons.append("uid_missing")
    uid_source = "onnx_uid_digits" if uid_from_image else ("uid_candidates" if uid else "missing")

    icon_agents, icon_reasons = _agents_from_icons(session_context)
    low_conf_reasons.extend(icon_reasons)
    pixel_discs_by_agent, pixel_disc_reasons = _pixel_discs_from_captures(session_context)
    low_conf_reasons.extend(pixel_disc_reasons)

    raw_agents = session_context.get("agents")
    if not isinstance(raw_agents, list):
        raw_agents = []
    parsed_agents, parsed_reasons = _extract_agents(raw_agents, session_context)
    low_conf_reasons.extend(parsed_reasons)

    if parsed_agents and icon_agents and len(parsed_agents) != len(icon_agents):
        low_conf_reasons.append("agent_count_mismatch_between_icon_and_payload")

    agents = _merge_agents(parsed_agents, icon_agents)
    agents, used_pixel_discs = _enrich_agents_with_pixel_discs(agents, pixel_discs_by_agent)
    if not agents:
        low_conf_reasons.append("agents_missing")

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
    screen_captures = session_context.get("screenCaptures")
    has_roster_capture = isinstance(screen_captures, list) and any(
        isinstance(capture, dict) and str(capture.get("role", "")).strip() == "roster"
        for capture in screen_captures
    )
    has_direct_crops = (
        bool(session_context.get("uidImagePath"))
        or bool(session_context.get("agentIconPaths"))
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
            if parsed_agents and icon_agents
            else ("onnx_agent_icon" if icon_agents else ("session_payload" if parsed_agents else "missing"))
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
        "agentIdsFromPixels": bool(icon_agents),
        "equipmentFromPixels": bool(used_pixel_discs),
        "fullRosterCoverage": False,
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
    context = {
        "sessionId": seed,
        "inputLockActive": True,
        "regionHint": region,
        "anchors": {"profile": True, "agents": True, "equipment": True},
        "uidCandidates": [uid_sample],
        "agents": agents if full_sync else agents[:2],
    }
    calibration = {"requiredAnchors": ["profile", "agents", "equipment"]}
    return scan_roster(context, calibration, locale="EN", resolution="1080p")
