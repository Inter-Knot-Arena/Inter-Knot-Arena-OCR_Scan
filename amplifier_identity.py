from __future__ import annotations

import difflib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import cv2

from equipment_taxonomy import display_name_for_id, normalize_name_and_id


ROOT = Path(__file__).resolve().parent
WIN_OCR_SCRIPT = ROOT / "scripts" / "win_ocr_batch.ps1"
TITLE_ALIAS_CONTRACT_PATH = ROOT / "contracts" / "amplifier-title-aliases.json"
TITLE_CROP_BOX = (0.28, 0.10, 0.66, 0.36)
INFO_CROP_BOX = (0.27, 0.07, 0.54, 0.53)
EFFECT_CROP_BOX = (0.30, 0.42, 0.63, 0.72)

EN_LANGUAGE_TAG = "en-US"
RU_LANGUAGE_TAG = "ru"

_DETAIL_FIELD_EMPTY_VALUES = (None, "", [], {})
_BASE_MARKERS = ("base stat", "базовые параметры")
_ADVANCED_MARKERS = ("advanced stat", "продвинутые параметры")
_EFFECT_MARKERS = ("w engine effect", "wengine effect", "w-engine effect", "эффект амплификатора")
_BASE_STAT_ALIASES = (
    ("attack_flat", ("base atk", "base attack", "базовая сила атаки", "сила атаки", "atk", "attack", "атк")),
    ("hp_flat", ("base hp", "базовое hp", "базовое нр", "базовое hp", "hp", "нр")),
    ("defense_flat", ("base def", "base defense", "базовая защита", "def", "defense", "защита")),
)
_ADVANCED_STAT_ALIASES = (
    ("impact", ("impact", "импульс")),
    ("crit_rate_pct", ("crit rate", "crit. rate", "шанс крит", "шанс крит попадания")),
    ("crit_damage_pct", ("crit dmg", "crit damage", "крит урон")),
    ("hp_pct", (" hp ", "hp%", " hp%", "нр", "hp")),
    ("attack_pct", ("atk%", "attack%", "attack", "atk", "сила атаки")),
    ("defense_pct", ("def%", "defense%", "defense", "def", "защита")),
    ("energy_regen", ("energy regen", "восстановление энергии")),
    ("pen_ratio_pct", ("pen ratio", "pen%", "процент пробивания", "пробивание")),
    ("pen_flat", ("pen ",)),
    ("anomaly_mastery", ("anomaly mastery", "контроль аномалии")),
    ("anomaly_proficiency", ("anomaly proficiency", "знание аномалии")),
)


@dataclass(slots=True)
class AmplifierDetailReadout:
    identity: AmplifierIdentityPrediction
    level: int | None = None
    level_cap: int | None = None
    base_stat_key: str | None = None
    base_stat_value: int | float | None = None
    advanced_stat_key: str | None = None
    advanced_stat_value: int | float | None = None
    info_text: str = ""
    effect_text: str = ""


@dataclass(slots=True)
class AmplifierIdentityPrediction:
    weapon_id: str
    display_name: str
    confidence: float
    source_mode: str
    raw_text: str
    title_candidate: str


def _text(value: object) -> str:
    return str(value or "").strip()


def _slugify(value: str) -> str:
    token = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    token = token.replace("&", " and ")
    token = token.replace("/", " ")
    token = token.replace("-", " ")
    token = re.sub(r"[^0-9a-zA-Z]+", " ", token)
    return "_".join(part for part in token.split() if part)


def normalize_alias_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value).lower())
    normalized = normalized.replace("\u2013", "-")
    normalized = normalized.replace("\u2014", "-")
    normalized = normalized.replace("\u2212", "-")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"(?i)\b(?:lvl?|lv\.?|ур\.?)\s*[0-9oiIl/ ]+\b", " ", normalized)
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"(?i)\b(?:base stat(?:s)?|базов(?:ые|ый)\s+параметр(?:ы)?|main stat|sub stats?)\b.*$", " ", normalized)
    normalized = re.sub(r"^[^0-9a-zа-яё]+", " ", normalized)
    normalized = re.sub(r"^(?:[0-9]{1,3}\s+)+", "", normalized)
    normalized = re.sub(r"(?i)^(?:[оo0]+)\s+", "", normalized)
    normalized = re.sub(r"\bnew\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(?:mew|нем)\b", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^[^0-9a-zа-яё]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" -")
    return normalized


def language_tag_for_locale(locale: str) -> str:
    return EN_LANGUAGE_TAG if str(locale or "").strip().upper() == "EN" else RU_LANGUAGE_TAG


def _crop_fractional_image(
    source_path: Path,
    output_path: Path,
    box: tuple[float, float, float, float],
    *,
    grayscale: bool,
    scale: int,
) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {source_path}")
    height, width = image.shape[:2]
    x0 = int(width * box[0])
    y0 = int(height * box[1])
    x1 = int(width * box[2])
    y1 = int(height * box[3])
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Empty crop: {source_path}")
    if grayscale:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
    if scale > 1:
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop):
        raise ValueError(f"Failed to write crop: {output_path}")


def crop_title_image(source_path: Path, output_path: Path) -> None:
    _crop_fractional_image(source_path, output_path, TITLE_CROP_BOX, grayscale=True, scale=2)


def crop_info_image(source_path: Path, output_path: Path) -> None:
    _crop_fractional_image(source_path, output_path, INFO_CROP_BOX, grayscale=False, scale=2)


def crop_effect_image(source_path: Path, output_path: Path) -> None:
    _crop_fractional_image(source_path, output_path, EFFECT_CROP_BOX, grayscale=True, scale=2)


def run_winrt_ocr_batch(crops: Sequence[Dict[str, str]], *, language_tag: str, temp_root: Path) -> Dict[str, str]:
    if not crops:
        return {}
    input_path = temp_root / f"ocr_input_{language_tag}.json"
    output_path = temp_root / f"ocr_output_{language_tag}.json"
    input_path.write_text(json.dumps(list(crops), ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WIN_OCR_SCRIPT),
            "-InputJson",
            str(input_path),
            "-OutputJson",
            str(output_path),
            "-LanguageTag",
            language_tag,
        ],
        check=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload_items = [payload]
    elif isinstance(payload, list):
        payload_items = payload
    else:
        payload_items = []
    return {
        str(item.get("id") or ""): _text(item.get("text"))
        for item in payload_items
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def extract_amplifier_title_candidate(text: str) -> str:
    normalized = " ".join(_text(text).split())
    if not normalized:
        return ""
    tokens = normalized.split()

    def _clean_token(token: str) -> str:
        return token.strip().strip("[](){}<>!,:;.|")

    def _looks_noise_prefix(token: str) -> bool:
        cleaned = _clean_token(token).lower()
        if not cleaned:
            return True
        if re.fullmatch(r"[0-9]{1,3}", cleaned):
            return True
        return cleaned in {"o", "о", "0", "new", "mew", "нем"}

    def _looks_noise_suffix(token: str) -> bool:
        cleaned = _clean_token(token).lower()
        if not cleaned:
            return True
        if cleaned in {"i", "ii", "iii", "iv", "v", "vi"}:
            return False
        if re.fullmatch(r"[0-9oiIl/]{2,}", cleaned):
            return True
        return cleaned in {"o", "о", "0", "e", "е"}

    def _looks_level_marker(token: str) -> bool:
        cleaned = _clean_token(token).lower()
        if not cleaned:
            return False
        if cleaned in {"ур", "yp", "lv", "lvl"}:
            return True
        if re.fullmatch(r"(?:ур|yp|lv|lvl)\.?", cleaned):
            return True
        return bool(re.fullmatch(r"[0-9oiIl/]{4,}", cleaned))

    def _looks_section_marker(token: str) -> bool:
        cleaned = _clean_token(token).lower()
        if not cleaned:
            return False
        return (
            cleaned.startswith("base")
            or cleaned.startswith("базов")
            or cleaned.startswith("парамет")
            or cleaned in {"main", "sub", "stat", "stats"}
        )

    while tokens and _looks_noise_prefix(tokens[0]):
        tokens.pop(0)

    collected: list[str] = []
    for token in tokens:
        if _looks_section_marker(token) or _looks_level_marker(token):
            break
        collected.append(token)

    while collected and _looks_noise_suffix(collected[-1]):
        collected.pop()

    candidate = " ".join(collected)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -")
    return candidate


@lru_cache(maxsize=1)
def title_alias_index() -> Dict[str, dict]:
    if not TITLE_ALIAS_CONTRACT_PATH.exists():
        return {}
    payload = json.loads(TITLE_ALIAS_CONTRACT_PATH.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    index: Dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized_alias = normalize_alias_key(str(entry.get("normalizedAlias") or entry.get("alias") or ""))
        amplifier_id = _text(entry.get("amplifierId"))
        display_name = _text(entry.get("displayName") or display_name_for_id(amplifier_id))
        if not normalized_alias or not amplifier_id or not display_name:
            continue
        if normalized_alias not in index:
            index[normalized_alias] = {
                "amplifierId": amplifier_id,
                "displayName": display_name,
            }
    return index


def _lookup_alias(candidate: str) -> Tuple[str, str, str]:
    normalized_candidate = normalize_alias_key(candidate)
    if not normalized_candidate:
        return "", "", ""

    index = title_alias_index()
    exact = index.get(normalized_candidate)
    if isinstance(exact, dict):
        return _text(exact.get("amplifierId")), _text(exact.get("displayName")), "alias_contract"

    best_key = ""
    best_entry: dict | None = None
    best_score = 0.0
    second_best = 0.0
    for key, entry in index.items():
        score = difflib.SequenceMatcher(None, normalized_candidate, key).ratio()
        if score > best_score:
            second_best = best_score
            best_score = score
            best_key = key
            best_entry = entry
        elif score > second_best:
            second_best = score
    if best_entry and best_score >= 0.90 and (best_score - second_best) >= 0.04:
        return _text(best_entry.get("amplifierId")), _text(best_entry.get("displayName")), "alias_contract_fuzzy"
    return "", "", ""


def confidence_for_source_mode(source_mode: str) -> float:
    normalized = _text(source_mode)
    if normalized == "alias_contract":
        return 0.995
    if normalized == "alias_contract_fuzzy":
        return 0.96
    if normalized == "taxonomy_direct":
        return 0.93
    if normalized == "generated_slug":
        return 0.78
    return 0.0


def classify_amplifier_text(text: str) -> AmplifierIdentityPrediction | None:
    raw_text = _text(text)
    title_candidate = extract_amplifier_title_candidate(raw_text)
    if not title_candidate:
        return None

    weapon_id, display_name, source_mode = _lookup_alias(title_candidate)
    if not weapon_id:
        try:
            weapon_id, display_name = normalize_name_and_id(title_candidate, "", kind="amplifier")
            source_mode = "taxonomy_direct"
        except ValueError:
            slug = _slugify(title_candidate)
            if not slug:
                return None
            weapon_id = f"amp_{slug}"
            display_name = title_candidate
            source_mode = "generated_slug"

    return AmplifierIdentityPrediction(
        weapon_id=weapon_id,
        display_name=display_name,
        confidence=confidence_for_source_mode(source_mode),
        source_mode=source_mode,
        raw_text=raw_text,
        title_candidate=title_candidate,
    )


def _normalize_detail_text(value: str) -> str:
    token = unicodedata.normalize("NFKC", _text(value).lower())
    token = token.replace("\u2013", "-")
    token = token.replace("\u2014", "-")
    token = token.replace("\u2212", "-")
    token = token.replace("\u00a0", " ")
    token = token.replace("5tat", "stat")
    token = token.replace("baso", "base")
    token = re.sub(r"([0-9]+(?:[.,][0-9]+)?)0/0\b", r"\1%", token)
    token = re.sub(r"([0-9]+(?:[.,][0-9]+)?)/0\b", r"\1%", token)
    token = token.replace(",", ".")
    token = re.sub(r"(?<!\d)\.(?!\d)", " ", token)
    token = re.sub(r"[^0-9a-zA-Z\u0400-\u04FF.%/+ ]+", " ", token)
    token = " ".join(token.split())
    return token


def _segment_between_markers(text: str, start_markers: Sequence[str], end_markers: Sequence[str]) -> str:
    normalized = _normalize_detail_text(text)
    if not normalized:
        return ""

    start_index = -1
    matched_start = ""
    for marker in start_markers:
        idx = normalized.find(marker)
        if idx >= 0 and (start_index < 0 or idx < start_index):
            start_index = idx
            matched_start = marker
    if start_index < 0:
        return ""

    segment_start = start_index + len(matched_start)
    end_index = len(normalized)
    for marker in end_markers:
        idx = normalized.find(marker, segment_start)
        if idx >= 0:
            end_index = min(end_index, idx)
    return normalized[segment_start:end_index].strip()


def _normalize_numeric_token(value: str) -> str:
    token = unicodedata.normalize("NFKC", _text(value).upper())
    token = token.replace("O", "0")
    token = token.replace("I", "1")
    token = token.replace("L", "1")
    token = token.replace("|", "/")
    token = token.replace("\\", "/")
    token = token.replace(",", ".")
    token = token.replace(" ", "")
    token = re.sub(r"([0-9]+(?:\.[0-9]+)?)0/0\b", r"\1%", token)
    token = re.sub(r"([0-9]+(?:\.[0-9]+)?)/0\b", r"\1%", token)
    return token


def _parse_level_pair(texts: Sequence[str]) -> tuple[int | None, int | None]:
    for text in texts:
        normalized = _normalize_detail_text(text)
        if not normalized:
            continue
        match = re.search(r"(?:\blv\.?|\bур\.?)\s*([0-9oOilIl/]{3,8})", normalized, flags=re.IGNORECASE)
        if not match:
            continue
        token = _normalize_numeric_token(match.group(1))
        left_digits = ""
        right_digits = ""
        if "/" in token:
            left_raw, right_raw = token.split("/", 1)
            left_digits = "".join(ch for ch in left_raw if ch.isdigit())
            right_digits = "".join(ch for ch in right_raw if ch.isdigit())
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if len(digits) == 4:
                left_digits, right_digits = digits[:2], digits[2:]
            elif len(digits) == 5:
                left_digits, right_digits = digits[:2], digits[-2:]
            elif len(digits) == 6:
                half = len(digits) // 2
                left_digits, right_digits = digits[:half], digits[half:]
        if not left_digits or not right_digits:
            continue
        current = int(left_digits)
        level_cap = int(right_digits)
        if 0 <= current <= 99 and 1 <= level_cap <= 99 and current <= level_cap:
            return current, level_cap
    return None, None


def _extract_value_tokens(text: str) -> list[str]:
    normalized = _normalize_detail_text(text)
    if not normalized:
        return []
    normalized = re.sub(r"(?:\blv\.?|\bур\.?)\s*[0-9oOilIl/]{3,8}", " ", normalized, flags=re.IGNORECASE)
    return re.findall(r"[+\-]?\d+(?:\.\d+)?%?", normalized)


def _coerce_numeric_value(token: str) -> int | float | None:
    normalized = _normalize_numeric_token(token)
    if not normalized:
        return None
    is_percent = normalized.endswith("%")
    if is_percent:
        normalized = normalized[:-1]
    if not normalized:
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    if not is_percent and value.is_integer():
        return int(value)
    return round(value, 4)


def _match_stat_key(segment: str, aliases: Sequence[tuple[str, Sequence[str]]]) -> str | None:
    normalized = f" {_normalize_detail_text(segment)} "
    if not normalized.strip():
        return None
    for stat_key, keys in aliases:
        for alias in keys:
            alias_normalized = f" {_normalize_detail_text(alias)} "
            if alias_normalized.strip() and alias_normalized in normalized:
                return stat_key
    return None


def parse_amplifier_detail(
    title_text: str,
    *,
    info_text: str = "",
    effect_text: str = "",
) -> AmplifierDetailReadout | None:
    identity = classify_amplifier_text(title_text)
    if identity is None:
        return None

    level, level_cap = _parse_level_pair((title_text, info_text, effect_text))
    info_segment_text = info_text or title_text
    base_segment = _segment_between_markers(info_segment_text, _BASE_MARKERS, (*_ADVANCED_MARKERS, *_EFFECT_MARKERS))
    advanced_segment = _segment_between_markers(
        info_segment_text,
        _ADVANCED_MARKERS,
        _EFFECT_MARKERS,
    )

    value_tokens = _extract_value_tokens(info_segment_text)
    numeric_values = [_coerce_numeric_value(token) for token in value_tokens]
    numeric_values = [value for value in numeric_values if value not in _DETAIL_FIELD_EMPTY_VALUES]

    base_stat_key = _match_stat_key(base_segment, _BASE_STAT_ALIASES)
    base_stat_value = numeric_values[0] if len(numeric_values) >= 1 else None
    if base_stat_key is None and base_stat_value is not None:
        # W-Engine base stat is ATK in every reviewed sample and runtime crop.
        base_stat_key = "attack_flat"

    advanced_stat_key = _match_stat_key(advanced_segment, _ADVANCED_STAT_ALIASES)
    advanced_stat_value = numeric_values[1] if len(numeric_values) >= 2 else None
    if advanced_stat_key is not None and isinstance(advanced_stat_value, (int, float)):
        if advanced_stat_key.endswith("_flat") and isinstance(advanced_stat_value, float) and advanced_stat_value.is_integer():
            advanced_stat_value = int(advanced_stat_value)

    return AmplifierDetailReadout(
        identity=identity,
        level=level,
        level_cap=level_cap,
        base_stat_key=base_stat_key,
        base_stat_value=base_stat_value,
        advanced_stat_key=advanced_stat_key,
        advanced_stat_value=advanced_stat_value,
        info_text=_text(info_text),
        effect_text=_text(effect_text),
    )
