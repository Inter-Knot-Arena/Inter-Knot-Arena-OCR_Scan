from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

from .model_runtime import ModelRegistry, classify_agent_icon, classify_uid_digits, get_model_metadata

SUPPORTED_LOCALES = {"RU", "EN"}
SUPPORTED_RESOLUTIONS = {"1080p", "1440p"}
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

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def _normalize_locale(locale: str | None) -> str:
    value = (locale or "EN").strip().upper()
    return value if value in SUPPORTED_LOCALES else ""


def _normalize_resolution(resolution: str | None) -> str:
    value = (resolution or "1080p").strip().lower()
    return value if value in SUPPORTED_RESOLUTIONS else ""


def _normalize_region(region_hint: str | None) -> str:
    value = (region_hint or "OTHER").strip().upper()
    return value if value in SUPPORTED_REGIONS else "OTHER"


def _digits_only(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _split_uid_digits(image: np.ndarray) -> list[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        3,
    )
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    h, w = threshold.shape[:2]
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < 18:
            continue
        if ch < int(h * 0.35):
            continue
        if cw > int(w * 0.4):
            continue
        boxes.append((x, y, cw, ch))

    boxes.sort(key=lambda item: item[0])
    digits: list[np.ndarray] = []
    for x, y, cw, ch in boxes:
        crop = threshold[y : y + ch, x : x + cw]
        digits.append(crop)
    return digits


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

    digit_images = _split_uid_digits(image)
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
        "mindscape": None,
        "weapon": {},
        "discs": [],
        "confidenceByField": {
            "agentId": round(confidence, 4),
            "level": 0.62,
            "mindscape": 0.62,
            "weapon": 0.58,
            "discs": 0.58,
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


def _extract_agents(payload: Iterable[Dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = []
    low_conf_reasons: list[str] = []

    for index, raw in enumerate(payload):
        agent_id = str(raw.get("agentId", "")).strip()
        if not agent_id:
            low_conf_reasons.append(f"agent[{index}].agentId_missing")
            continue

        level_raw = raw.get("level")
        level = float(level_raw) if isinstance(level_raw, (int, float)) else None
        level_conf = 0.84 if level is None else (0.88 + 0.12 * _normalized_fraction(level, 1.0, 60.0))
        if level is None:
            low_conf_reasons.append(f"{agent_id}.level_missing")

        mindscape_raw = raw.get("mindscape")
        mindscape = float(mindscape_raw) if isinstance(mindscape_raw, (int, float)) else None
        mindscape_conf = 0.82 if mindscape is None else (0.88 + 0.1 * _normalized_fraction(mindscape, 0.0, 6.0))
        if mindscape is None:
            low_conf_reasons.append(f"{agent_id}.mindscape_missing")

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

        confidence_by_field = {
            "agentId": float(raw.get("agentConfidence", 0.99)),
            "level": round(level_conf, 4),
            "mindscape": round(mindscape_conf, 4),
            "weapon": round(weapon_conf, 4),
            "discs": round(disc_conf, 4),
        }

        results.append(
            {
                "agentId": agent_id,
                "level": level,
                "mindscape": mindscape,
                "weapon": weapon,
                "discs": discs,
                "confidenceByField": confidence_by_field,
            }
        )

    if not results:
        low_conf_reasons.append("agents_empty_after_parse")

    return results, low_conf_reasons


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
    normalized_resolution = _normalize_resolution(resolution)
    if not normalized_locale or not normalized_resolution:
        raise ScanFailure(
            ScanFailureCode.UNSUPPORTED_LAYOUT,
            "Unsupported locale or resolution.",
            [
                f"locale={locale}",
                f"resolution={resolution}",
                "supported_locales=RU|EN",
                "supported_resolutions=1080p|1440p",
            ],
        )

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

    icon_agents, icon_reasons = _agents_from_icons(session_context)
    low_conf_reasons.extend(icon_reasons)

    raw_agents = session_context.get("agents")
    if not isinstance(raw_agents, list):
        raw_agents = []
    parsed_agents, parsed_reasons = _extract_agents(raw_agents)
    low_conf_reasons.extend(parsed_reasons)

    agents = icon_agents if icon_agents else parsed_agents
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
        )

    return result


def run_scan(seed: str, region: str, full_sync: bool) -> Dict[str, Any]:
    rng = random.Random(seed)
    agents = [
        {
            "agentId": "agent_anby",
            "level": rng.randint(40, 60),
            "mindscape": rng.randint(0, 6),
            "weapon": {"weaponId": "amp_starlight_engine", "level": rng.randint(40, 60)},
            "discs": [
                {"slot": 1, "setId": "set_woodpecker", "level": rng.randint(12, 15)},
                {"slot": 2, "setId": "set_woodpecker", "level": rng.randint(12, 15)},
            ],
        },
        {
            "agentId": "agent_nicole",
            "level": rng.randint(40, 60),
            "mindscape": rng.randint(0, 6),
            "weapon": {"weaponId": "amp_rainforest", "level": rng.randint(40, 60)},
            "discs": [
                {"slot": 1, "setId": "set_swing_jazz", "level": rng.randint(12, 15)},
                {"slot": 2, "setId": "set_swing_jazz", "level": rng.randint(12, 15)},
            ],
        },
        {
            "agentId": "agent_ellen",
            "level": rng.randint(40, 60),
            "mindscape": rng.randint(0, 6),
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
