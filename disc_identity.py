from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from equipment_taxonomy import display_name_for_id, known_disc_set_ids, normalize_display_text, normalize_name_and_id


RU_DISC_SET_ALIASES = {
    "\u0434\u044f\u0442\u043b\u043e\u043a\u043e\u0440 \u044d\u043b\u0435\u043a\u0442\u0440\u043e": "set_woodpecker_electro",
    "\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u044f \u044e\u043d\u044c\u043a\u0443\u0439": "set_yunkui_tales",
    "\u0448\u043e\u043a\u0441\u0442\u0430\u0440 \u0434\u0438\u0441\u043a\u043e": "set_shockstar_disco",
    "\u043f\u0435\u0441\u043d\u044c \u043e \u0432\u0435\u0442\u043a\u0435 \u0438 \u043a\u043b\u0438\u043d\u043a\u0435": "set_branch_and_blade_song",
    "\u0446\u0432\u0435\u0442\u043e\u043a \u043d\u0430 \u0440\u0430\u0441\u0441\u0432\u0435\u0442\u0435": "set_dawn_s_bloom",
    "\u0432\u043b\u0430\u0434\u044b\u043a\u0430 \u0433\u043e\u0440\u044b": "set_king_of_the_summit",
    "\u0433\u0430\u0440\u043c\u043e\u043d\u0438\u044f \u0442\u0435\u043d\u0435\u0439": "set_shadow_harmony",
    "\u0433\u0440\u043e\u0437\u043e\u0432\u043e\u0439 \u0445\u044d\u0432\u0438 \u043c\u0435\u0442\u0430\u043b\u043b": "set_thunder_metal",
    "\u043b\u0443\u0447\u0435\u0437\u0430\u0440\u043d\u0430\u044f \u0430\u0440\u0438\u044f": "set_shining_aria",
    "\u0433\u043e\u0440\u043c\u043e\u043d \u043f\u0430\u043d\u043a": "set_hormone_punk",
    "\u0431\u0430\u043b\u043b\u0430\u0434\u0430 \u043e \u0444\u0430\u044d\u0442\u043e\u043d\u0435": "set_phaethon_s_melody",
    "\u0441\u0432\u0438\u043d\u0433 \u0434\u0436\u0430\u0437": "set_swing_jazz",
    "\u0441\u043e\u0443\u043b \u0440\u043e\u043a": "set_soul_rock",
    "\u0441\u0432\u0438\u0440\u0435\u043f\u044b\u0439 \u0445\u044d\u0432\u0438 \u043c\u0435\u0442\u0430\u043b\u043b": "set_fanged_metal",
    "\u0444\u0443\u0433\u0443 \u044d\u043b\u0435\u043a\u0442\u0440\u043e": "set_puffer_electro",
    "\u043f\u0440\u043e\u0442\u043e\u043f\u0430\u043d\u043a": "set_proto_punk",
    "\u0445\u0430\u043e\u0441 \u0434\u0436\u0430\u0437": "set_chaos_jazz",
    "\u0445\u0430\u043e\u0441 \u043c\u0435\u0442\u0430\u043b\u043b": "set_chaotic_metal",
    "\u043f\u043e\u043b\u044f\u0440\u043d\u044b\u0439 \u0445\u044d\u0432\u0438 \u043c\u0435\u0442\u0430\u043b\u043b": "set_polar_metal",
    "\u0430\u0441\u0442\u0440\u0430\u043b\u044c\u043d\u044b\u0439 \u0433\u043e\u043b\u043e\u0441": "set_astral_voice",
    "\u043f\u0435\u0441\u043d\u044c \u043e \u0441\u0438\u043d\u0438\u0445 \u0432\u043e\u0434\u0430\u0445": "set_white_water_ballad",
    "\u0441\u0432\u043e\u0431\u043e\u0434\u0430 \u0431\u043b\u044e\u0437": "set_freedom_blues",
    "\u0444\u0440\u0438\u0434\u043e\u043c \u0431\u043b\u044e\u0437": "set_freedom_blues",
    "\u0438\u043d\u0444\u0435\u0440\u043d\u043e \u043c\u0435\u0442\u0430\u043b\u043b": "set_inferno_metal",
    "\u043b\u0443\u043d\u043d\u0430\u044f \u043a\u043e\u043b\u044b\u0431\u0435\u043b\u044c\u043d\u0430\u044f": "set_moonlight_lullaby",
    "\u043f\u043e\u043b\u044f\u0440\u043d\u044b\u0439 \u043c\u0435\u0442\u0430\u043b\u043b": "set_polar_metal",
    "\u0433\u0440\u043e\u0437\u043e\u0432\u043e\u0439 \u043c\u0435\u0442\u0430\u043b\u043b": "set_thunder_metal",
    "\u0441\u0432\u0438\u0440\u0435\u043f\u044b\u0439 \u043c\u0435\u0442\u0430\u043b\u043b": "set_fanged_metal",
    "\u0448\u043e\u043a\u0441\u0442\u0430\u0440-\u0434\u0438\u0441\u043a\u043e": "set_shockstar_disco",
    "\u0441\u0432\u0438\u0440\u0435\u043f\u044b\u0439 \u0445\u044d\u0432\u0438-\u043c\u0435\u0442\u0430\u043b": "set_fanged_metal",
    "\u0441\u0432\u0438\u043d\u0433-\u0434\u0436\u0430\u0437": "set_swing_jazz",
    "\u043f\u043e\u043b\u044f\u0440\u043d\u044b\u0439 \u0445\u044d\u0432\u0438-\u043c\u0435\u0442\u0430\u043b": "set_polar_metal",
    "\u0444\u0440\u0438\u0434\u043e\u043c-\u0431\u043b\u044e\u0437": "set_freedom_blues",
    "\u0433\u0440\u043e\u0437\u043e\u0432\u043e\u0439 \u0445\u044d\u0432\u0438-\u043c\u0435\u0442\u0430\u043b": "set_thunder_metal",
    "\u0434\u044f\u0442\u043b\u043e\u043a\u043e\u0440-\u044d\u043b\u0435\u043a\u0442\u0440\u043e": "set_woodpecker_electro",
    "\u0445\u0430\u043e\u0441-\u0434\u0436\u0430\u0437": "set_chaos_jazz",
    "\u0445\u0430\u043e\u0441-\u043c\u0435\u0442\u0430\u043b": "set_chaotic_metal",
    "\u0433\u043e\u0440\u043c\u043e\u043d-\u043f\u0430\u043d\u043a": "set_hormone_punk",
    "\u0444\u0443\u0433\u0443-\u044d\u043b\u0435\u043a\u0442\u0440\u043e": "set_puffer_electro",
    "\u0438\u043d\u0444\u0435\u0440\u043d\u043e-\u043c\u0435\u0442\u0430\u043b": "set_inferno_metal",
    "\u0441\u043e\u0443\u043b-\u0440\u043e\u043a": "set_soul_rock",
    "\u044f\u0442\u043b\u043e\u043a\u043e\u0440 \u044d\u043b\u0435\u043a\u0442\u0440\u043e": "set_woodpecker_electro",
    "woodpecker eiectro": "set_woodpecker_electro",
    "soui rock": "set_soul_rock",
    "astrai voice": "set_astral_voice",
}


@dataclass(slots=True)
class DiscTitlePrediction:
    set_id: str
    display_name: str
    confidence: float
    source_mode: str
    raw_text: str
    title_candidate: str


def _text(value: object) -> str:
    return str(value or "").strip()


def _normalize_alias_key(value: str) -> str:
    normalized = normalize_display_text(value)
    normalized = normalized.replace("\u0451", "\u0435")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\bnew\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^[\W_]+", " ", normalized)
    normalized = re.sub(r"^[a-z\u0430-\u044f]\s+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^[0-9]+\s+", " ", normalized)
    normalized = re.sub(r"^\b[\u043eo]\b\s*", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(?:\u0443\u0440|yp|lv)[a-z\u0430-\u044f0-9/.-]*\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bo\b$", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\u043e\b$", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b[\u0435e]\b$", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalized_mapping_lookup(mapping: Mapping[str, str], candidate: str) -> str:
    normalized_candidate = _normalize_alias_key(candidate)
    if not normalized_candidate:
        return ""
    for raw_key, mapped_value in mapping.items():
        if _normalize_alias_key(str(raw_key)) == normalized_candidate:
            return str(mapped_value)
    return ""


def _fuzzy_lookup(mapping: Mapping[str, str], candidate: str, *, threshold: float, min_margin: float = 0.05) -> Tuple[str, float]:
    normalized_candidate = _normalize_alias_key(candidate)
    if not normalized_candidate:
        return "", 0.0
    best_value = ""
    best_score = 0.0
    second_best = 0.0
    for raw_key, mapped_value in mapping.items():
        normalized_key = _normalize_alias_key(str(raw_key))
        if not normalized_key:
            continue
        score = difflib.SequenceMatcher(None, normalized_candidate, normalized_key).ratio()
        if score > best_score:
            second_best = best_score
            best_score = score
            best_value = str(mapped_value)
        elif score > second_best:
            second_best = score
    if best_value and best_score >= threshold and (best_score - second_best) >= min_margin:
        return best_value, best_score
    return "", 0.0


def extract_disk_title_candidate(text: str) -> str:
    normalized = " ".join(_text(text).split())
    if not normalized:
        return ""
    candidate = re.split(
        r"(?i)\b(?:lv(?:\.|[a-z0-9/.-]+)?|\u0443\u0440(?:\.|[a-z\u0430-\u044f0-9/.-]+)?|main stat|sub-?stats|[a-z\u0430-\u044f0-9]+\s+stat|\u0431\u0430\u0437\u043e\u0432(?:\u044b\u0439|\u044b\u0435)|\u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b)\b",
        normalized,
    )[0]
    candidate = re.sub(r"\[[^\]]*\]", " ", candidate)
    candidate = re.sub(r"\[[^\]]*$", " ", candidate)
    candidate = re.sub(r"\bnew\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^[\W_]+", " ", candidate)
    candidate = re.sub(r"^[a-z\u0430-\u044f]\s+", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^[0-9]+\s+", " ", candidate)
    candidate = re.sub(r"^\b[\u043eo]\b\s*", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b[0-9]{1,2}\b", " ", candidate)
    candidate = re.sub(r"\b(?:\u0443\u0440|yp|lv)[a-z\u0430-\u044f0-9/.-]*\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\bo\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b\u043e\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b[\u0435e]\b$", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -")
    return candidate


def confidence_for_disc_source_mode(source_mode: str) -> float:
    normalized = _text(source_mode)
    if normalized == "ru_alias":
        return 0.96
    if normalized == "ru_alias_fuzzy":
        return 0.9
    if normalized == "taxonomy_direct":
        return 0.88
    if normalized == "taxonomy_fuzzy":
        return 0.8
    return 0.0


def classify_disc_title(text: str) -> DiscTitlePrediction | None:
    raw_text = _text(text)
    title_candidate = extract_disk_title_candidate(raw_text)
    if not title_candidate:
        return None

    set_id = _normalized_mapping_lookup(RU_DISC_SET_ALIASES, title_candidate)
    source_mode = "ru_alias"
    if not set_id:
        set_id, _ = _fuzzy_lookup(RU_DISC_SET_ALIASES, title_candidate, threshold=0.84, min_margin=0.0)
        source_mode = "ru_alias_fuzzy"
    if set_id:
        return DiscTitlePrediction(
            set_id=set_id,
            display_name=display_name_for_id(set_id),
            confidence=confidence_for_disc_source_mode(source_mode),
            source_mode=source_mode,
            raw_text=raw_text,
            title_candidate=title_candidate,
        )

    try:
        set_id, display_name = normalize_name_and_id(title_candidate, "", kind="disc_set")
        return DiscTitlePrediction(
            set_id=set_id,
            display_name=display_name,
            confidence=confidence_for_disc_source_mode("taxonomy_direct"),
            source_mode="taxonomy_direct",
            raw_text=raw_text,
            title_candidate=title_candidate,
        )
    except ValueError:
        pass

    best_id = ""
    best_display = ""
    best_score = 0.0
    normalized_candidate = _normalize_alias_key(title_candidate)
    for known_id in sorted(str(item) for item in known_disc_set_ids()):
        display_name = display_name_for_id(known_id)
        if not display_name:
            continue
        score = difflib.SequenceMatcher(None, normalized_candidate, _normalize_alias_key(display_name)).ratio()
        if score > best_score:
            best_id = known_id
            best_display = display_name
            best_score = score
    if best_id and best_score >= 0.72:
        return DiscTitlePrediction(
            set_id=best_id,
            display_name=best_display,
            confidence=confidence_for_disc_source_mode("taxonomy_fuzzy"),
            source_mode="taxonomy_fuzzy",
            raw_text=raw_text,
            title_candidate=title_candidate,
        )
    return None
