from __future__ import annotations

import os
import tempfile
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import cv2
import numpy as np

from ocr_dataset_policy import (
    AGENT_DETAIL_ROLE,
    AMPLIFIER_DETAIL_ROLE,
    DISK_DETAIL_ROLE,
    EQUIPMENT_ROLE,
    ROSTER_ROLE,
    UID_PANEL_ROLE,
)


_CANONICAL_RESOLUTIONS = ("1080p", "1440p")
_TARGET_ASPECT_RATIO = 16.0 / 9.0
_ASPECT_TOLERANCE = 0.2

# Canonical layout boxes are stored as fractions so the same runtime path can
# crop from any captured frame that maps to the supported layout family.
_UID_PANEL_UID_BOX = (0.02, 0.095, 0.19, 0.062)
_UID_PANEL_UID_BOXES = (
    _UID_PANEL_UID_BOX,
    (0.02, 0.155, 0.28, 0.10),
    (0.02, 0.17, 0.24, 0.07),
)
_AGENT_DETAIL_UID_BOXES = (
    (0.848, 0.929, 0.148, 0.069),
    (0.842, 0.924, 0.158, 0.075),
    (0.859, 0.938, 0.137, 0.058),
)
_UID_PANEL_DIGITS_BOX = (0.355, 0.555, 0.37, 0.28)
_UID_WIDGET_THRESHOLDS = (200, 180, 160, 140, 120, 100, 80, 60, 40)
_UID_WIDGET_COMPONENT_MIN_HEIGHT = 10
_UID_WIDGET_COMPONENT_MAX_WIDTH = 64
_UID_WIDGET_COMPONENT_MIN_AREA = 8
_UID_WIDGET_BAND_TOLERANCE = 18.0
_ROSTER_AGENT_ICON_BOXES = (
    (0.527, 0.014, 0.160, 0.194),
    (0.566, 0.222, 0.133, 0.208),
    (0.605, 0.479, 0.141, 0.278),
)
_ROSTER_PAGE_CAPACITY = len(_ROSTER_AGENT_ICON_BOXES)
_AGENT_IDENTITY_BOXES = {
    AGENT_DETAIL_ROLE: (
        ("portrait_wide", (0.02, 0.08, 0.46, 0.84)),
        ("portrait_tight", (0.10, 0.10, 0.32, 0.62)),
    ),
    EQUIPMENT_ROLE: (
        ("portrait_wide", (0.02, 0.06, 0.48, 0.88)),
        ("portrait_tight", (0.08, 0.08, 0.34, 0.66)),
    ),
    AMPLIFIER_DETAIL_ROLE: (
        ("portrait_wide", (0.02, 0.06, 0.48, 0.88)),
        ("portrait_tight", (0.08, 0.08, 0.34, 0.66)),
    ),
    DISK_DETAIL_ROLE: (
        ("portrait_wide", (0.02, 0.06, 0.48, 0.88)),
        ("portrait_tight", (0.08, 0.08, 0.34, 0.66)),
    ),
}


@dataclass(slots=True)
class RuntimeCapture:
    role: str
    path: Path
    alias: str
    agent_id: str
    slot_index: int | None = None
    agent_slot_index: int | None = None
    page_index: int | None = None


def _parse_resolution_height(value: str) -> int | None:
    text = str(value or "").strip().lower()
    if text.endswith("p") and text[:-1].isdigit():
        return int(text[:-1])
    if "x" in text:
        parts = text.split("x", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[1])
    return None


def _canonical_resolution_for_height(height: int) -> str:
    return "1440p" if int(height) >= 1296 else "1080p"


def _is_layout_aspect(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    ratio = float(width) / float(height)
    return abs(ratio - _TARGET_ASPECT_RATIO) <= _ASPECT_TOLERANCE


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _effective_page_index(page_index: int | None, fallback: int = 0) -> int:
    if isinstance(page_index, int) and page_index >= 0:
        return page_index
    return fallback


def _effective_agent_slot_index(agent_slot_index: int | None, page_index: int | None) -> int | None:
    if agent_slot_index is None or agent_slot_index <= 0:
        return None
    return agent_slot_index


def _load_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _capture_shape(session_context: Mapping[str, Any]) -> tuple[int, int] | None:
    raw_captures = session_context.get("screenCaptures")
    if isinstance(raw_captures, list):
        preferred = sorted(
            (
                entry for entry in raw_captures
                if isinstance(entry, dict) and _as_text(entry.get("path"))
            ),
            key=lambda entry: 0 if _as_text(entry.get("role")) == ROSTER_ROLE else 1,
        )
        for entry in preferred:
            image = _load_image(Path(_as_text(entry.get("path"))))
            if image is not None:
                height, width = image.shape[:2]
                return width, height
    return None


def normalize_runtime_resolution(session_context: Mapping[str, Any], resolution: str | None) -> str:
    explicit = str(resolution or "").strip().lower()
    explicit_canonical = ""
    if explicit in _CANONICAL_RESOLUTIONS:
        explicit_canonical = explicit

    explicit_height = _parse_resolution_height(explicit)
    if explicit_height is not None:
        explicit_canonical = _canonical_resolution_for_height(explicit_height)

    capture_shape = _capture_shape(session_context)
    if capture_shape is not None:
        width, height = capture_shape
        if not _is_layout_aspect(width, height):
            return ""
        observed_canonical = _canonical_resolution_for_height(height)
        if explicit_canonical and explicit_canonical != observed_canonical:
            return ""
        return explicit_canonical or observed_canonical

    return explicit_canonical


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


def crop_agent_identity_candidates(image: np.ndarray, role: str) -> list[tuple[str, np.ndarray]]:
    normalized_role = _as_text(role)
    variants = _AGENT_IDENTITY_BOXES.get(normalized_role)
    if not variants:
        return []
    output: list[tuple[str, np.ndarray]] = []
    for variant_name, box in variants:
        crop = _fractional_crop(image, box)
        if crop is None:
            continue
        output.append((variant_name, crop))
    return output


def _component_bands(
    components: List[tuple[int, int, int, int, int, float]]
) -> List[List[tuple[int, int, int, int, int, float]]]:
    ordered = sorted(components, key=lambda item: item[5])
    bands: List[List[tuple[int, int, int, int, int, float]]] = []
    current: List[tuple[int, int, int, int, int, float]] = []
    for component in ordered:
        if not current or abs(component[5] - current[-1][5]) <= _UID_WIDGET_BAND_TOLERANCE:
            current.append(component)
            continue
        bands.append(current)
        current = [component]
    if current:
        bands.append(current)
    return bands


def _extract_uid_digit_band(widget_crop: np.ndarray) -> tuple[np.ndarray | None, float]:
    gray = widget_crop if widget_crop.ndim == 2 else cv2.cvtColor(widget_crop, cv2.COLOR_BGR2GRAY)
    best_band: np.ndarray | None = None
    best_score: float | None = None

    for threshold in _UID_WIDGET_THRESHOLDS:
        mask = (gray >= threshold).astype(np.uint8)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components: List[tuple[int, int, int, int, int, float]] = []
        for component_index in range(1, component_count):
            x, y, width, height, area = stats[component_index]
            if area < _UID_WIDGET_COMPONENT_MIN_AREA:
                continue
            if height < _UID_WIDGET_COMPONENT_MIN_HEIGHT:
                continue
            if width > _UID_WIDGET_COMPONENT_MAX_WIDTH:
                continue
            components.append((int(x), int(y), int(width), int(height), int(area), float(y + height / 2.0)))
        if len(components) < 10:
            continue

        for band in _component_bands(components):
            band = sorted(band, key=lambda item: item[0])
            if len(band) < 10:
                continue
            for start_index in range(0, len(band) - 9):
                sequence = band[start_index : start_index + 10]
                gaps = [sequence[idx + 1][0] - (sequence[idx][0] + sequence[idx][2]) for idx in range(9)]
                widths = [item[2] for item in sequence]
                heights = [item[3] for item in sequence]
                if max(gaps) > 28 or min(gaps) < -2:
                    continue
                prefix_count = start_index
                suffix_count = len(band) - (start_index + 10)
                score = 0.0
                score -= statistics.pstdev(gaps) * 4.0
                score -= statistics.pstdev(widths) * 1.2
                score -= statistics.pstdev(heights) * 1.0
                score += sum(widths) * 0.15
                if 3 <= prefix_count <= 5:
                    score += 16.0
                else:
                    score -= abs(prefix_count - 4) * 4.0
                if 0 <= suffix_count <= 2:
                    score += 8.0
                else:
                    score -= abs(suffix_count - 1) * 3.0

                x0 = max(0, min(item[0] for item in sequence) - 2)
                y0 = max(0, min(item[1] for item in sequence) - 2)
                x1 = min(widget_crop.shape[1], max(item[0] + item[2] for item in sequence) + 2)
                y1 = min(widget_crop.shape[0], max(item[1] + item[3] for item in sequence) + 2)
                band_crop = widget_crop[y0:y1, x0:x1]
                if band_crop.size == 0:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_band = band_crop.copy()

    return best_band, float(best_score or 0.0)


def crop_uid_region(image: np.ndarray, role: str) -> np.ndarray | None:
    normalized_role = _as_text(role)
    if normalized_role == UID_PANEL_ROLE:
        best_crop: np.ndarray | None = None
        best_score = float("-inf")
        for box in _UID_PANEL_UID_BOXES:
            widget_crop = _fractional_crop(image, box)
            if widget_crop is None:
                continue
            digit_band, score = _extract_uid_digit_band(widget_crop)
            if digit_band is not None and score > best_score:
                best_score = score
                best_crop = digit_band
                continue
            digits_crop = _fractional_crop(widget_crop, _UID_PANEL_DIGITS_BOX)
            if digits_crop is not None and score > best_score:
                best_score = score
                best_crop = digits_crop
        return best_crop
    if normalized_role in {AGENT_DETAIL_ROLE, EQUIPMENT_ROLE}:
        best_crop: np.ndarray | None = None
        best_score = float("-inf")
        for box in _AGENT_DETAIL_UID_BOXES:
            widget_crop = _fractional_crop(image, box)
            if widget_crop is None:
                continue
            digit_band, score = _extract_uid_digit_band(widget_crop)
            if digit_band is not None and score > best_score:
                best_score = score
                best_crop = digit_band.copy()
                continue
            if score > best_score:
                best_score = score
                best_crop = widget_crop.copy()
        return best_crop
    return None


def _ensure_temp_root(session_id: str) -> Path:
    override_root = str(os.environ.get("IKA_RUNTIME_TEMP_ROOT") or "").strip()
    if override_root:
        root = Path(override_root) / "ika_ocr_runtime" / session_id
    else:
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
            slot_index = _as_index(entry.get("slotIndex"))
            page_index = _as_index(entry.get("pageIndex"))
            agent_slot_index = _effective_agent_slot_index(_as_index(entry.get("agentSlotIndex")), page_index)
            captures.append(
                RuntimeCapture(
                    role=role,
                    path=Path(path_value),
                    alias=_as_text(entry.get("screenAlias") or entry.get("alias") or f"{role}_{index}"),
                    agent_id=_as_text(entry.get("agentId") or entry.get("focusAgentId")),
                    slot_index=slot_index,
                    agent_slot_index=agent_slot_index,
                    page_index=page_index,
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
                alias = _as_text(entry.get("screenAlias") or entry.get("alias") or f"legacy_agent_icon_{index + 1}")
                page_index = _as_index(entry.get("pageIndex"))
                agent_slot_index = _as_index(entry.get("agentSlotIndex"))
            else:
                path_value = _as_text(entry)
                alias = f"legacy_agent_icon_{index + 1}"
                page_index = None
                agent_slot_index = None
            if not path_value:
                continue
            captures.append(
                RuntimeCapture(
                    role="agent_icon_crop",
                    path=Path(path_value),
                    alias=alias,
                    agent_id="",
                    agent_slot_index=agent_slot_index,
                    page_index=page_index,
                )
            )
    return captures


def _derive_uid_from_capture(capture: RuntimeCapture, root: Path) -> str | None:
    if capture.role not in {UID_PANEL_ROLE, AGENT_DETAIL_ROLE, EQUIPMENT_ROLE}:
        return None
    image = _load_image(capture.path)
    if image is None:
        return None
    crop = crop_uid_region(image, capture.role)
    if crop is None:
        return None
    return _save_crop(root, "uid_from_uid_panel.png", crop)


def _derive_agent_icons_from_capture(
    capture: RuntimeCapture,
    root: Path,
    *,
    page_index: int,
) -> List[Dict[str, Any]]:
    if capture.role != ROSTER_ROLE:
        return []
    image = _load_image(capture.path)
    if image is None:
        return []

    icons: List[Dict[str, Any]] = []
    for index, box in enumerate(_ROSTER_AGENT_ICON_BOXES):
        crop = _fractional_crop(image, box)
        if crop is None:
            continue
        path = _save_crop(root, f"agent_icon_page_{page_index}_{index + 1}.png", crop)
        if path:
            icons.append(
                {
                    "path": path,
                    "screenAlias": capture.alias,
                    "pageIndex": page_index,
                    "agentSlotIndex": (page_index * _ROSTER_PAGE_CAPACITY) + index + 1,
                    "rosterPageSlotIndex": index + 1,
                }
            )
    return icons


def normalize_runtime_captures(session_context: Dict[str, Any], resolution: str | None) -> Dict[str, Any]:
    normalized = dict(session_context)
    captures = _collect_runtime_captures(session_context)
    if not captures:
        return normalized

    normalized_resolution = normalize_runtime_resolution(session_context, resolution)
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
        roster_captures = [
            (index, capture)
            for index, capture in enumerate(captures)
            if capture.role == ROSTER_ROLE
        ]
        derived_icons: list[dict[str, Any]] = []
        for derived_page_index, (_, capture) in enumerate(
            sorted(
                roster_captures,
                key=lambda item: (
                    _effective_page_index(item[1].page_index, item[0]),
                    item[0],
                ),
            )
        ):
            effective_page_index = _effective_page_index(capture.page_index, derived_page_index)
            derived_icons.extend(
                _derive_agent_icons_from_capture(
                    capture,
                    temp_root,
                    page_index=effective_page_index,
                )
            )
        if derived_icons:
            normalized["agentIconPaths"] = sorted(
                derived_icons,
                key=lambda entry: (
                    _as_index(entry.get("pageIndex")) or 0,
                    _as_index(entry.get("agentSlotIndex")) or 0,
                    _as_text(entry.get("path")),
                ),
            )

    anchors_raw = normalized.get("anchors")
    normalized["anchors"] = dict(anchors_raw) if isinstance(anchors_raw, dict) else {}
    normalized["runtimeResolution"] = normalized_resolution
    normalized["screenCaptures"] = [
        {
            "role": capture.role,
            "path": str(capture.path),
            "screenAlias": capture.alias,
            "agentId": capture.agent_id,
            "slotIndex": capture.slot_index,
            "agentSlotIndex": capture.agent_slot_index,
            "pageIndex": capture.page_index,
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
