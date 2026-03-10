from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import cv2
import numpy as np

from ocr_dataset_policy import ROSTER_ROLE, UID_PANEL_ROLE


_SUPPORTED_RESOLUTIONS = {"1080p", "1440p"}

# Canonical layout boxes are stored as fractions so the same runtime path can
# crop from any captured frame that maps to the supported layout family.
_UID_PANEL_UID_BOX = (0.02, 0.095, 0.19, 0.062)
_ROSTER_UID_BOX = (0.603, 0.852, 0.336, 0.124)
_ROSTER_AGENT_ICON_BOXES = (
    (0.612, 0.139, 0.272, 0.207),
    (0.612, 0.365, 0.272, 0.207),
    (0.612, 0.591, 0.272, 0.207),
)


@dataclass(slots=True)
class RuntimeCapture:
    role: str
    path: Path
    alias: str
    agent_id: str
    slot_index: int | None = None


def _normalize_resolution(resolution: str | None) -> str:
    value = (resolution or "1080p").strip().lower()
    return value if value in _SUPPORTED_RESOLUTIONS else "1080p"


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _load_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _fractional_crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    x, y, w, h = box
    x0 = max(0, min(int(round(width * x)), width - 1))
    y0 = max(0, min(int(round(height * y)), height - 1))
    x1 = max(x0 + 1, min(int(round(width * (x + w))), width))
    y1 = max(y0 + 1, min(int(round(height * (y + h))), height))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def _ensure_temp_root(session_id: str) -> Path:
    root = Path(tempfile.gettempdir()) / "ika_ocr_runtime" / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_crop(root: Path, stem: str, image: np.ndarray) -> str | None:
    path = root / stem
    if cv2.imwrite(str(path), image):
        return str(path)
    return None


def _collect_runtime_captures(session_context: Mapping[str, Any]) -> List[RuntimeCapture]:
    captures: List[RuntimeCapture] = []
    raw_captures = session_context.get("screenCaptures")
    if isinstance(raw_captures, list):
        for index, entry in enumerate(raw_captures):
            if not isinstance(entry, dict):
                continue
            role = _as_text(entry.get("role") or entry.get("screenRole"))
            path_value = _as_text(entry.get("path"))
            if not role or not path_value:
                continue
            slot_index_raw = entry.get("slotIndex")
            slot_index = int(slot_index_raw) if isinstance(slot_index_raw, int) else None
            captures.append(
                RuntimeCapture(
                    role=role,
                    path=Path(path_value),
                    alias=_as_text(entry.get("screenAlias") or entry.get("alias") or f"{role}_{index}"),
                    agent_id=_as_text(entry.get("agentId") or entry.get("focusAgentId")),
                    slot_index=slot_index,
                )
            )

    uid_image_path = _as_text(session_context.get("uidImagePath"))
    if uid_image_path:
        captures.append(
            RuntimeCapture(
                role=UID_PANEL_ROLE,
                path=Path(uid_image_path),
                alias="legacy_uid_crop",
                agent_id="",
            )
        )

    raw_icons = session_context.get("agentIconPaths")
    if isinstance(raw_icons, list):
        for index, entry in enumerate(raw_icons):
            if isinstance(entry, dict):
                path_value = _as_text(entry.get("path"))
            else:
                path_value = _as_text(entry)
            if not path_value:
                continue
            captures.append(
                RuntimeCapture(
                    role="agent_icon_crop",
                    path=Path(path_value),
                    alias=f"legacy_agent_icon_{index + 1}",
                    agent_id="",
                )
            )
    return captures


def _derive_uid_from_capture(capture: RuntimeCapture, root: Path) -> str | None:
    if capture.role == UID_PANEL_ROLE:
        image = _load_image(capture.path)
        if image is None:
            return None
        crop = _fractional_crop(image, _UID_PANEL_UID_BOX)
        if crop is None:
            return None
        return _save_crop(root, "uid_from_uid_panel.png", crop)

    if capture.role == ROSTER_ROLE:
        image = _load_image(capture.path)
        if image is None:
            return None
        crop = _fractional_crop(image, _ROSTER_UID_BOX)
        if crop is None:
            return None
        return _save_crop(root, "uid_from_roster.png", crop)

    return None


def _derive_agent_icons_from_capture(capture: RuntimeCapture, root: Path) -> List[Dict[str, str]]:
    if capture.role != ROSTER_ROLE:
        return []
    image = _load_image(capture.path)
    if image is None:
        return []

    icons: List[Dict[str, str]] = []
    for index, box in enumerate(_ROSTER_AGENT_ICON_BOXES):
        crop = _fractional_crop(image, box)
        if crop is None:
            continue
        path = _save_crop(root, f"agent_icon_{index + 1}.png", crop)
        if path:
            icons.append({"path": path, "screenAlias": capture.alias})
    return icons


def normalize_runtime_captures(session_context: Dict[str, Any], resolution: str | None) -> Dict[str, Any]:
    normalized = dict(session_context)
    captures = _collect_runtime_captures(session_context)
    if not captures:
        return normalized

    normalized_resolution = _normalize_resolution(resolution)
    session_id = _as_text(session_context.get("sessionId")) or "session"
    temp_root = _ensure_temp_root(session_id)

    if not _as_text(normalized.get("uidImagePath")):
        for capture in captures:
            uid_path = _derive_uid_from_capture(capture, temp_root)
            if uid_path:
                normalized["uidImagePath"] = uid_path
                break

    raw_icon_payload = normalized.get("agentIconPaths")
    has_agent_icons = isinstance(raw_icon_payload, list) and bool(raw_icon_payload)
    if not has_agent_icons:
        for capture in captures:
            icons = _derive_agent_icons_from_capture(capture, temp_root)
            if icons:
                normalized["agentIconPaths"] = icons
                break

    anchors_raw = normalized.get("anchors")
    anchors = dict(anchors_raw) if isinstance(anchors_raw, dict) else {}
    if _as_text(normalized.get("uidImagePath")):
        anchors["profile"] = True
    icon_payload = normalized.get("agentIconPaths")
    if isinstance(icon_payload, list) and len(icon_payload) >= 2:
        anchors["agents"] = True
        anchors.setdefault("equipment", True)
    normalized["anchors"] = anchors
    normalized["runtimeResolution"] = normalized_resolution
    normalized["screenCaptures"] = [
        {
            "role": capture.role,
            "path": str(capture.path),
            "screenAlias": capture.alias,
            "agentId": capture.agent_id,
            "slotIndex": capture.slot_index,
        }
        for capture in captures
    ]
    return normalized


def derive_equipment_occupancy(
    *,
    weapon: Mapping[str, Any] | None,
    discs: Iterable[Mapping[str, Any]] | None,
    explicit_weapon_present: Any = None,
    explicit_slot_occupancy: Any = None,
) -> Dict[str, Any]:
    slot_occupancy = {str(slot): False for slot in range(1, 7)}
    if isinstance(explicit_slot_occupancy, dict):
        for key, value in explicit_slot_occupancy.items():
            slot_key = str(key).strip()
            if slot_key in slot_occupancy:
                slot_occupancy[slot_key] = bool(value)

    if discs:
        for disc in discs:
            if not isinstance(disc, Mapping):
                continue
            slot_raw = disc.get("slot")
            if isinstance(slot_raw, (int, float)):
                slot_key = str(int(slot_raw))
            else:
                slot_key = _as_text(slot_raw)
            if slot_key in slot_occupancy:
                slot_occupancy[slot_key] = True

    weapon_present: bool | None
    if explicit_weapon_present is not None:
        weapon_present = bool(explicit_weapon_present)
    elif isinstance(weapon, Mapping):
        weapon_present = any(
            value not in (None, "", [], {})
            for key, value in weapon.items()
            if key not in {"agentId"}
        )
    else:
        weapon_present = False

    return {
        "weaponPresent": weapon_present,
        "discSlotOccupancy": slot_occupancy,
    }
