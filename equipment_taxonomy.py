from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Tuple

AMPLIFIER_ALIASES: Dict[str, tuple[str, ...]] = {
    "amp_deep_sea_visitor": (
        "amp_deep_sea_visitor",
        "amp_deep_sea",
        "deep sea visitor",
        "guest from the deep",
        "гость из глубин",
    ),
}

DISC_SET_ALIASES: Dict[str, tuple[str, ...]] = {
    "set_woodpecker_electro": (
        "set_woodpecker_electro",
        "woodpecker electro",
        "дятлокор-электро",
        "дятлокор электро",
    ),
    "set_branch_and_blade_song": (
        "set_branch_and_blade_song",
        "branch and blade song",
        "song of branch and blade",
        "песнь о ветке и клинке",
    ),
}

DISPLAY_NAMES: Dict[str, str] = {
    "amp_deep_sea_visitor": "Гость из глубин",
    "set_woodpecker_electro": "Дятлокор-электро",
    "set_branch_and_blade_song": "Песнь о ветке и клинке",
}

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


def normalize_display_text(value: str) -> str:
    token = str(value or "").strip().lower()
    token = token.replace("ё", "е")
    token = token.replace("–", "-")
    token = token.replace("—", "-")
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
    return (question_count / len(meaningful)) >= 0.35


def _alias_index(alias_map: Dict[str, tuple[str, ...]]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for canonical, aliases in alias_map.items():
        for raw in aliases:
            normalized = normalize_display_text(raw)
            if normalized:
                output[normalized] = canonical
    return output


@lru_cache(maxsize=1)
def _amplifier_index() -> Dict[str, str]:
    return _alias_index(AMPLIFIER_ALIASES)


@lru_cache(maxsize=1)
def _disc_set_index() -> Dict[str, str]:
    return _alias_index(DISC_SET_ALIASES)


def canonicalize_amplifier_name(value: str) -> str:
    return _amplifier_index().get(normalize_display_text(value), "")


def canonicalize_disc_set_name(value: str) -> str:
    return _disc_set_index().get(normalize_display_text(value), "")


def display_name_for_id(canonical_id: str) -> str:
    return DISPLAY_NAMES.get(str(canonical_id or "").strip(), "")


def valid_stat_key(value: str) -> bool:
    return str(value or "").strip() in VALID_STAT_KEYS


def normalize_name_and_id(
    raw_name: str,
    raw_id: str,
    *,
    kind: str,
) -> Tuple[str, str]:
    requested_id = str(raw_id or "").strip()
    requested_name = str(raw_name or "").strip()
    if kind == "amplifier":
        canonical = requested_id or canonicalize_amplifier_name(requested_name)
    elif kind == "disc_set":
        canonical = requested_id or canonicalize_disc_set_name(requested_name)
    else:
        raise ValueError(f"Unsupported taxonomy kind: {kind}")
    if not canonical:
        raise ValueError(f"Invalid {kind} value: id={requested_id!r}, name={requested_name!r}")
    display_name = requested_name
    if looks_corrupted_display_name(display_name):
        display_name = ""
    if not display_name:
        display_name = display_name_for_id(canonical)
    if not display_name:
        raise ValueError(f"Missing display name for {kind}: {canonical}")
    return canonical, display_name


def known_amplifier_ids() -> Iterable[str]:
    return AMPLIFIER_ALIASES.keys()


def known_disc_set_ids() -> Iterable[str]:
    return DISC_SET_ALIASES.keys()
