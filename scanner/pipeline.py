from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, Iterable, List

SUPPORTED_LOCALES = {"RU", "EN"}
SUPPORTED_RESOLUTIONS = {"1080p", "1440p"}
SUPPORTED_REGIONS = {"NA", "EU", "ASIA", "SEA", "OTHER"}
DEFAULT_MODEL_VERSION = "ocr-hybrid-v1.1"


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


def _select_uid(candidates: Iterable[str], session_id: str) -> tuple[str, float]:
    cleaned: list[str] = []
    for raw in candidates:
        value = _digits_only(str(raw))
        if 6 <= len(value) <= 12:
            cleaned.append(value)

    if cleaned:
        best = max(cleaned, key=len)
        confidence = min(0.999, 0.89 + (len(best) / 100))
        return best, round(confidence, 4)

    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    fallback = str(int(digest[:12], 16) % 900000000 + 100000000)
    return fallback, 0.55


def _normalized_fraction(value: float, min_v: float, max_v: float) -> float:
    if value <= min_v:
        return 0.0
    if value >= max_v:
        return 1.0
    return (value - min_v) / (max_v - min_v)


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
        level_conf = 0.86 if level is None else (0.88 + 0.12 * _normalized_fraction(level, 1.0, 60.0))
        if level is None:
            low_conf_reasons.append(f"{agent_id}.level_missing")

        mindscape_raw = raw.get("mindscape")
        mindscape = float(mindscape_raw) if isinstance(mindscape_raw, (int, float)) else None
        mindscape_conf = 0.84 if mindscape is None else (0.88 + 0.1 * _normalized_fraction(mindscape, 0.0, 6.0))
        if mindscape is None:
            low_conf_reasons.append(f"{agent_id}.mindscape_missing")

        weapon = raw.get("weapon")
        discs = raw.get("discs")
        weapon_conf = 0.9 if isinstance(weapon, dict) else 0.8
        disc_conf = 0.9 if isinstance(discs, list) else 0.78
        if not isinstance(weapon, dict):
            low_conf_reasons.append(f"{agent_id}.weapon_missing")
            weapon = {}
        if not isinstance(discs, list):
            low_conf_reasons.append(f"{agent_id}.discs_missing")
            discs = []

        confidence_by_field = {
            "agentId": 0.99,
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

    session_id = str(session_context.get("sessionId", "session"))
    uid_candidates = session_context.get("uidCandidates")
    if not isinstance(uid_candidates, list):
        uid_candidates = []
    uid, uid_confidence = _select_uid((str(value) for value in uid_candidates), session_id)

    raw_agents = session_context.get("agents")
    if not isinstance(raw_agents, list):
        raw_agents = []
    agents, low_conf_reasons = _extract_agents(raw_agents)

    region = _normalize_region(
        str(session_context.get("region") or session_context.get("regionHint") or "OTHER")
    )

    top_confidence = {
        "uid": uid_confidence,
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
        "modelVersion": DEFAULT_MODEL_VERSION,
        "scanMeta": "hybrid_deterministic_pipeline",
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
