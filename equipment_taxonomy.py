from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Tuple


ROOT = Path(__file__).resolve().parent
TAXONOMY_PATH = ROOT / "contracts" / "equipment-taxonomy.json"

VALID_STAT_KEYS = {
    "anomaly_mastery",
    "anomaly_proficiency",
    "attack_flat",
    "attack_pct",
    "crit_damage_pct",
    "crit_rate_pct",
    "defense_flat",
    "defense_pct",
    "energy_regen",
    "hp_flat",
    "hp_pct",
    "ice_damage_bonus_pct",
    "impact",
    "pen_flat",
    "pen_ratio_pct",
}


def _strip_optional_suffixes(value: str) -> str:
    text = value
    text = re.sub(r"\s*\[[1-6]\]\s*$", "", text)
    text = re.sub(r"\s*\((?:slot|disc|disk)\s*[1-6]\)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*slot\s*[1-6]\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_display_text(value: str) -> str:
    token = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    token = token.replace("\u2013", "-")
    token = token.replace("\u2014", "-")
    token = token.replace("\u2212", "-")
    token = token.replace("\u00a0", " ")
    token = token.replace("&", " and ")
    token = token.replace("/", " ")
    token = token.replace("_", " ")
    token = _strip_optional_suffixes(token)
    token = token.replace("w-engine", "w engine")
    token = token.replace("w engine", "wengine")
    token = re.sub(r"[^0-9a-zA-Z\u0400-\u04FF+\- ]+", " ", token)
    token = " ".join(token.split())
    return token


def looks_corrupted_display_name(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return True
    meaningful = [char for char in token if not char.isspace() and char not in "-_[]()"]
    if not meaningful:
        return True
    question_count = sum(1 for char in meaningful if char == "?")
    if question_count == len(meaningful):
        return True
    mojibake_markers = ("Р", "Ѓ", "Є", "ё", "ї", "ј")
    marker_hits = sum(1 for char in meaningful if char in mojibake_markers)
    if marker_hits and (marker_hits / len(meaningful)) >= 0.35:
        return True
    return (question_count / len(meaningful)) >= 0.35


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _kind_key(kind: str) -> str:
    if kind == "amplifier":
        return "amplifiers"
    if kind == "disc_set":
        return "discSets"
    raise ValueError(f"Unsupported taxonomy kind: {kind}")


def _id_field(kind: str) -> str:
    if kind == "amplifier":
        return "amplifierId"
    if kind == "disc_set":
        return "discSetId"
    raise ValueError(f"Unsupported taxonomy kind: {kind}")


@lru_cache(maxsize=1)
def _taxonomy_payload() -> dict:
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            "Equipment taxonomy snapshot is missing. "
            "Run `python scripts/sync_equipment_taxonomy.py` first."
        )
    return _read_json(TAXONOMY_PATH)


@lru_cache(maxsize=2)
def _taxonomy_index(kind: str) -> Dict[str, dict]:
    payload = _taxonomy_payload()
    collection = payload.get(_kind_key(kind))
    if not isinstance(collection, list):
        raise ValueError(f"Invalid taxonomy payload: {_kind_key(kind)} must be an array")

    identifier_key = _id_field(kind)
    index: Dict[str, dict] = {}
    for entry in collection:
        if not isinstance(entry, dict):
            continue
        canonical_id = str(entry.get(identifier_key) or "").strip()
        if not canonical_id:
            continue
        aliases = entry.get("aliases")
        if isinstance(aliases, list):
            raw_aliases = aliases
        else:
            raw_aliases = []
        for raw in [canonical_id, *raw_aliases]:
            normalized = normalize_display_text(str(raw))
            if normalized and normalized not in index:
                index[normalized] = entry
    return index


@lru_cache(maxsize=2)
def _display_name_index(kind: str) -> Dict[str, str]:
    payload = _taxonomy_payload()
    collection = payload.get(_kind_key(kind))
    if not isinstance(collection, list):
        return {}
    identifier_key = _id_field(kind)
    output: Dict[str, str] = {}
    for entry in collection:
        if not isinstance(entry, dict):
            continue
        canonical_id = str(entry.get(identifier_key) or "").strip()
        display_name = str(entry.get("displayName") or "").strip()
        if canonical_id and display_name:
            output[canonical_id] = display_name
    return output


def canonicalize_amplifier_name(value: str) -> str:
    entry = _taxonomy_index("amplifier").get(normalize_display_text(value))
    return str(entry.get("amplifierId") or "").strip() if isinstance(entry, dict) else ""


def canonicalize_disc_set_name(value: str) -> str:
    entry = _taxonomy_index("disc_set").get(normalize_display_text(value))
    return str(entry.get("discSetId") or "").strip() if isinstance(entry, dict) else ""


def display_name_for_id(canonical_id: str) -> str:
    token = str(canonical_id or "").strip()
    if not token:
        return ""
    for kind in ("amplifier", "disc_set"):
        display_name = _display_name_index(kind).get(token)
        if display_name:
            return display_name
    return ""


def valid_stat_key(value: str) -> bool:
    return str(value or "").strip() in VALID_STAT_KEYS


def normalize_name_and_id(
    raw_name: str,
    raw_id: str,
    *,
    kind: str,
) -> Tuple[str, str]:
    identifier_key = _id_field(kind)
    requested_id = str(raw_id or "").strip()
    requested_name = _strip_optional_suffixes(str(raw_name or "").strip())

    entry = None
    index = _taxonomy_index(kind)
    for candidate in (requested_id, requested_name):
        normalized = normalize_display_text(candidate)
        if normalized:
            entry = index.get(normalized)
        if isinstance(entry, dict):
            break

    if not isinstance(entry, dict):
        raise ValueError(f"Invalid {kind} value: id={requested_id!r}, name={requested_name!r}")

    canonical_id = str(entry.get(identifier_key) or "").strip()
    if not canonical_id:
        raise ValueError(f"Missing canonical {kind} id for entry: {entry!r}")

    display_name = requested_name
    if looks_corrupted_display_name(display_name):
        display_name = ""
    if not display_name:
        display_name = str(entry.get("displayName") or "").strip()
    if not display_name:
        raise ValueError(f"Missing display name for {kind}: {canonical_id}")
    return canonical_id, display_name


def known_amplifier_ids() -> Iterable[str]:
    return _display_name_index("amplifier").keys()


def known_disc_set_ids() -> Iterable[str]:
    return _display_name_index("disc_set").keys()
