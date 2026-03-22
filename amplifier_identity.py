from __future__ import annotations

import difflib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import cv2

from equipment_taxonomy import display_name_for_id, normalize_name_and_id


ROOT = Path(__file__).resolve().parent
WIN_OCR_SCRIPT = ROOT / "scripts" / "win_ocr_batch.ps1"
TITLE_ALIAS_CONTRACT_PATH = ROOT / "contracts" / "amplifier-title-aliases.json"
TITLE_CROP_BOX = (0.28, 0.10, 0.66, 0.36)
INFO_CROP_BOX = (0.27, 0.07, 0.54, 0.53)
ADVANCED_STAT_CROP_BOX = (0.28, 0.32, 0.50, 0.43)
ADVANCED_STAT_FALLBACK_CROP_BOX = (0.24, 0.28, 0.60, 0.47)
EFFECT_CROP_BOX = (0.30, 0.42, 0.63, 0.72)
_VALID_LEVEL_CAPS = {10, 20, 30, 40, 50, 60}

EN_LANGUAGE_TAG = "en-US"
RU_LANGUAGE_TAG = "ru"

_DETAIL_FIELD_EMPTY_VALUES = (None, "", [], {})
_BASE_MARKERS = ("base stat", "base stats", "базовые параметры")
_ADVANCED_MARKERS = ("advanced stat", "advanced stats", "продвинутые параметры")
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
    _crop_fractional_image(source_path, output_path, TITLE_CROP_BOX, grayscale=True, scale=1)


def crop_info_image(source_path: Path, output_path: Path) -> None:
    _crop_fractional_image(source_path, output_path, INFO_CROP_BOX, grayscale=False, scale=1)


def crop_advanced_stat_image(source_path: Path, output_path: Path) -> None:
    _crop_fractional_image(source_path, output_path, ADVANCED_STAT_CROP_BOX, grayscale=True, scale=3)


def crop_advanced_stat_fallback_image(source_path: Path, output_path: Path) -> None:
    _crop_fractional_image(
        source_path,
        output_path,
        ADVANCED_STAT_FALLBACK_CROP_BOX,
        grayscale=True,
        scale=4,
    )


def crop_effect_image(source_path: Path, output_path: Path) -> None:
    _crop_fractional_image(source_path, output_path, EFFECT_CROP_BOX, grayscale=True, scale=1)


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


def classify_amplifier_detail_identity(
    title_text: str,
    *,
    info_text: str = "",
) -> AmplifierIdentityPrediction | None:
    candidates = (
        (0, title_text),
        (1, info_text),
        (2, " ".join(part for part in (_text(title_text), _text(info_text)) if part)),
    )
    seen: set[str] = set()
    best_identity: AmplifierIdentityPrediction | None = None
    best_score: tuple[int, float, int, int] | None = None
    source_mode_priority = {
        "alias_contract": 3,
        "alias_contract_fuzzy": 2,
        "taxonomy_direct": 1,
        "generated_slug": 0,
    }
    for source_rank, candidate in candidates:
        normalized = _text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        identity = classify_amplifier_text(normalized)
        if identity is None:
            continue
        score = (
            source_mode_priority.get(identity.source_mode, -1),
            float(identity.confidence),
            -int(source_rank),
            len(identity.title_candidate),
        )
        if best_score is None or score > best_score:
            best_identity = identity
            best_score = score
    return best_identity


def _normalize_detail_text(value: str) -> str:
    token = unicodedata.normalize("NFKC", _text(value).lower())
    token = token.replace("\u2013", "-")
    token = token.replace("\u2014", "-")
    token = token.replace("\u2212", "-")
    token = token.replace("\u00a0", " ")
    token = token.replace("5tat", "stat")
    token = token.replace("baso", "base")
    token = token.replace(",", ".")
    token = re.sub(r"(?<!\d)\.(?!\d)", " ", token)
    token = re.sub(r"[^0-9a-zA-Z\u0400-\u04FF.%/+ ]+", " ", token)
    token = " ".join(token.split())
    return token


def _segment_between_markers(text: str, start_markers: Sequence[str], end_markers: Sequence[str]) -> str:
    normalized = _normalize_detail_text(text)
    if not normalized:
        return ""

    start_span = _find_marker_span(normalized, start_markers)
    if start_span is None:
        return ""
    segment_start = start_span[1]
    end_index = len(normalized)
    end_span = _find_marker_span(normalized, end_markers, start=segment_start)
    if end_span is not None:
        end_index = end_span[0]
    return normalized[segment_start:end_index].strip()


def _normalize_numeric_token(value: str) -> str:
    token = unicodedata.normalize("NFKC", _text(value).upper())
    token = token.replace("O", "0")
    token = token.replace("I", "1")
    token = token.replace("L", "1")
    token = token.replace("О", "0")
    token = token.replace("Л", "1")
    token = token.replace("І", "1")
    token = token.replace("|", "/")
    token = token.replace("\\", "/")
    token = token.replace(",", ".")
    token = token.replace(" ", "")
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
    return re.findall(r"[+\-]?\d+(?:\.\d+)?(?:/0)?%?", normalized)


def _coerce_slash_percent_value(token: str) -> float | None:
    normalized = _normalize_numeric_token(token)
    if not normalized.endswith("/0"):
        return None
    raw = normalized[:-2]
    if not raw:
        return None
    try:
        base_value = float(raw)
    except ValueError:
        return None
    percent_value = base_value if "." in raw else (base_value / 10.0)
    return round(percent_value, 4)


def _coerce_slash_base_value(token: str) -> int | None:
    normalized = _normalize_numeric_token(token)
    if not normalized.endswith("/0"):
        return None
    raw = normalized[:-2]
    if not raw or "." in raw or not raw.isdigit():
        return None
    return int(raw)


def _coerce_numeric_value(token: str) -> int | float | None:
    normalized = _normalize_numeric_token(token)
    if not normalized:
        return None
    slash_percent_value = _coerce_slash_percent_value(normalized)
    if slash_percent_value is not None:
        return slash_percent_value
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
            if not alias_normalized.strip():
                continue
            if alias_normalized in normalized:
                return stat_key
            alias_tokens = _normalize_detail_text(alias).split()
            if len(alias_tokens) < 2:
                continue
            pattern = r"\b" + re.escape(alias_tokens[0])
            for token in alias_tokens[1:]:
                pattern += r"(?:\s+(?:[0-9./%]+)\s+|\s+)" + re.escape(token)
            pattern += r"\b"
            if re.search(pattern, normalized):
                return stat_key
    return None


def _normalize_level_token(value: str) -> str:
    token = _normalize_numeric_token(value)
    token = token.replace("\u041E", "0")
    token = token.replace("\u043E", "0")
    token = token.replace("\u041B", "1")
    token = token.replace("\u043B", "1")
    token = token.replace("\u0406", "1")
    token = token.replace("\u0456", "1")
    return token


def _parse_level_token_robust(token: str) -> tuple[int | None, int | None]:
    normalized = _normalize_level_token(token)
    if not normalized:
        return None, None

    left_digits = ""
    right_digits = ""
    if "/" in normalized:
        left_raw, right_raw = normalized.split("/", 1)
        left_digits = "".join(ch for ch in left_raw if ch.isdigit())
        right_digits = "".join(ch for ch in right_raw if ch.isdigit())
    else:
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if len(digits) == 3 and digits[1:] in {str(value) for value in _VALID_LEVEL_CAPS}:
            left_digits = digits[0]
            right_digits = digits[1:]
            if left_digits == "0":
                left_digits = digits[1]
        elif len(digits) == 4:
            left_digits, right_digits = digits[:2], digits[2:]
        elif len(digits) == 5:
            left_digits, right_digits = digits[:2], digits[-2:]
        elif len(digits) == 6:
            half = len(digits) // 2
            left_digits, right_digits = digits[:half], digits[half:]

    if not left_digits or not right_digits:
        return None, None
    current = int(left_digits)
    level_cap = int(right_digits)
    if current <= 0 or level_cap not in _VALID_LEVEL_CAPS or current > level_cap:
        return None, None
    return current, level_cap


def _parse_level_pair_robust(texts: Sequence[str]) -> tuple[int | None, int | None]:
    explicit_pattern = re.compile(
        r"(?:\blv\.?|\b\u0443\u0440\.?)\s*([0-9oOilIl\u041E\u043E\u041B\u043B\u0406\u0456/]{3,8})",
        flags=re.IGNORECASE,
    )
    fallback_pattern = re.compile(r"\b([0-9oOilIl\u041E\u043E\u041B\u043B\u0406\u0456/]{3,8})\b", flags=re.IGNORECASE)
    for text in texts:
        normalized = _normalize_detail_text(text)
        if not normalized:
            continue
        for match in explicit_pattern.finditer(normalized):
            current, level_cap = _parse_level_token_robust(match.group(1))
            if current is not None and level_cap is not None:
                return current, level_cap
        for match in fallback_pattern.finditer(normalized[:64]):
            current, level_cap = _parse_level_token_robust(match.group(1))
            if current is not None and level_cap is not None:
                return current, level_cap
    return None, None


def _aliases_for_stat_key(
    stat_key: str | None,
    aliases: Sequence[tuple[str, Sequence[str]]],
) -> Sequence[str]:
    if not stat_key:
        return ()
    for alias_key, alias_values in aliases:
        if alias_key == stat_key:
            return alias_values
    return ()


def _extract_numeric_values(text: str) -> list[tuple[str, int | float]]:
    values: list[tuple[str, int | float]] = []
    for token in _extract_value_tokens(text):
        numeric = _coerce_numeric_value(token)
        if numeric in _DETAIL_FIELD_EMPTY_VALUES:
            continue
        values.append((token, numeric))
    return values


def _segment_after_alias(text: str, aliases: Sequence[str]) -> str:
    normalized = _normalize_detail_text(text)
    if not normalized:
        return ""

    best_index = -1
    matched_alias = ""
    for alias in aliases:
        alias_normalized = _normalize_detail_text(alias)
        if not alias_normalized:
            continue
        idx = normalized.find(alias_normalized)
        if idx >= 0 and (best_index < 0 or idx < best_index):
            best_index = idx
            matched_alias = alias_normalized
    if best_index < 0:
        return ""
    return normalized[best_index + len(matched_alias) :].strip()


def _find_marker_span(
    normalized_text: str,
    markers: Sequence[str],
    *,
    start: int = 0,
) -> tuple[int, int] | None:
    best_span: tuple[int, int] | None = None
    haystack = normalized_text[start:]
    if not haystack:
        return None

    for marker in markers:
        normalized_marker = _normalize_detail_text(marker)
        if not normalized_marker:
            continue
        pattern = (
            r"(?<![0-9a-zA-Z\u0400-\u04FF])"
            + re.escape(normalized_marker)
            + r"(?![0-9a-zA-Z\u0400-\u04FF])"
        )
        match = re.search(pattern, haystack)
        if match is not None:
            span = (start + match.start(), start + match.end())
        else:
            idx = haystack.find(normalized_marker)
            if idx < 0:
                continue
            span = (start + idx, start + idx + len(normalized_marker))
        if best_span is None or span[0] < best_span[0]:
            best_span = span
    return best_span


def _segment_after_markers(text: str, start_markers: Sequence[str]) -> str:
    normalized = _normalize_detail_text(text)
    if not normalized:
        return ""

    marker_span = _find_marker_span(normalized, start_markers)
    if marker_span is None:
        return ""
    return normalized[marker_span[1] :].strip()


def _numeric_equal(left: int | float | None, right: int | float | None) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return abs(float(left) - float(right)) < 0.05


def _normalize_stat_value_for_key(stat_key: str | None, value: int | float | None) -> int | float | None:
    if value in _DETAIL_FIELD_EMPTY_VALUES:
        return value
    if not stat_key or not isinstance(value, (int, float)):
        return value

    numeric = float(value)
    if stat_key == "impact":
        while numeric > 30.0:
            numeric /= 10.0
        return round(numeric, 4)
    if stat_key.endswith("_pct") or stat_key == "energy_regen":
        while numeric > 100.0:
            numeric /= 10.0
        return round(numeric, 4)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 4)


def _is_compatible_stat_value(stat_key: str | None, value: int | float | None) -> bool:
    if not stat_key or not isinstance(value, (int, float)):
        return False
    normalized = _normalize_stat_value_for_key(stat_key, value)
    if not isinstance(normalized, (int, float)):
        return False
    numeric = float(normalized)
    ranges: dict[str, tuple[float, float]] = {
        "attack_flat": (100.0, 5000.0),
        "hp_flat": (100.0, 50000.0),
        "defense_flat": (10.0, 5000.0),
        "impact": (1.0, 30.0),
        "crit_rate_pct": (0.0, 35.0),
        "crit_damage_pct": (0.0, 80.0),
        "hp_pct": (0.0, 40.0),
        "attack_pct": (0.0, 40.0),
        "defense_pct": (0.0, 40.0),
        "energy_regen": (0.0, 40.0),
        "pen_ratio_pct": (0.0, 40.0),
        "pen_flat": (0.0, 1000.0),
        "anomaly_mastery": (0.0, 200.0),
        "anomaly_proficiency": (0.0, 200.0),
    }
    lower, upper = ranges.get(stat_key, (0.0, 100000.0))
    return lower <= numeric <= upper


def _pick_first_compatible_value(
    stat_key: str | None,
    values: Sequence[int | float],
) -> int | float | None:
    for value in values:
        if _is_compatible_stat_value(stat_key, value):
            return value
    return None


def _normalize_token_value_for_stat_key(
    stat_key: str | None,
    token: str,
    value: int | float,
) -> int | float:
    normalized = _normalize_stat_value_for_key(stat_key, value)
    if (
        stat_key == "energy_regen"
        and isinstance(normalized, (int, float))
        and _normalize_numeric_token(token).endswith("/0")
    ):
        numeric = float(normalized)
        while numeric > 40.0:
            numeric /= 10.0
        normalized = round(numeric, 4)
    return normalized


def _pick_first_compatible_token_value(
    stat_key: str | None,
    values: Sequence[tuple[str, int | float]],
) -> int | float | None:
    for token, value in values:
        normalized = _normalize_token_value_for_stat_key(stat_key, token, value)
        if _is_compatible_stat_value(stat_key, normalized):
            return normalized
    return None


def _pick_best_base_value(
    stat_key: str | None,
    values: Sequence[int | float],
) -> tuple[int | float | None, int | None]:
    candidates = [
        (index, value)
        for index, value in enumerate(values)
        if _is_compatible_stat_value(stat_key, value)
    ]
    if candidates:
        index, value = max(candidates, key=lambda item: float(item[1]))
        return value, index
    return None, None


def _pick_first_distinct_value(
    values: Iterable[int | float],
    *,
    excluding: int | float | None,
) -> int | float | None:
    for value in values:
        if _numeric_equal(value, excluding):
            continue
        return value
    return None


def parse_amplifier_detail(
    title_text: str,
    *,
    info_text: str = "",
    advanced_text: str = "",
    effect_text: str = "",
) -> AmplifierDetailReadout | None:
    identity = classify_amplifier_detail_identity(
        title_text,
        info_text=info_text,
    )
    if identity is None:
        return None

    level, level_cap = _parse_level_pair_robust((title_text, info_text, effect_text))
    info_segment_text = info_text or title_text
    base_segment = _segment_between_markers(info_segment_text, _BASE_MARKERS, (*_ADVANCED_MARKERS, *_EFFECT_MARKERS))
    advanced_segment = _segment_between_markers(
        info_segment_text,
        _ADVANCED_MARKERS,
        _EFFECT_MARKERS,
    )
    advanced_text_segment = _segment_after_markers(advanced_text, _ADVANCED_MARKERS) or advanced_text

    base_stat_key = _match_stat_key(base_segment, _BASE_STAT_ALIASES)
    base_stat_value = None
    base_value_index: int | None = None

    advanced_stat_key = _match_stat_key(advanced_text_segment, _ADVANCED_STAT_ALIASES)
    if advanced_stat_key is None:
        advanced_stat_key = _match_stat_key(advanced_segment, _ADVANCED_STAT_ALIASES)
    title_base_values = [
        value
        for _, value in _extract_numeric_values(
            _segment_after_alias(title_text, _aliases_for_stat_key(base_stat_key, _BASE_STAT_ALIASES))
        )
    ]
    effect_section_values = [
        value
        for _, value in _extract_numeric_values(_segment_after_markers(info_segment_text, _EFFECT_MARKERS))
    ]
    effect_token_values = _extract_numeric_values(
        _segment_after_alias(effect_text, _aliases_for_stat_key(advanced_stat_key, _ADVANCED_STAT_ALIASES))
    )
    advanced_line_token_values = _extract_numeric_values(
        _segment_after_alias(advanced_text_segment, _aliases_for_stat_key(advanced_stat_key, _ADVANCED_STAT_ALIASES))
    )
    advanced_segment_token_values = _extract_numeric_values(
        _segment_after_alias(advanced_segment, _aliases_for_stat_key(advanced_stat_key, _ADVANCED_STAT_ALIASES))
    )
    ordered_token_values = _extract_numeric_values(info_segment_text)
    ordered_values = [value for _, value in ordered_token_values]
    ordered_slash_base_values = [
        value
        for token, _ in ordered_token_values
        for value in [_coerce_slash_base_value(token)]
        if value is not None
    ]

    if title_base_values:
        base_stat_value, base_value_index = _pick_best_base_value(base_stat_key, title_base_values)
        if base_stat_value is None:
            base_stat_value = title_base_values[0]
            base_value_index = 0
    if base_stat_value is None and len(effect_section_values) >= 2:
        base_stat_value = effect_section_values[0]
        base_value_index = next(
            (index for index, value in enumerate(ordered_values) if _numeric_equal(value, base_stat_value)),
            None,
        )
    if base_stat_value is None:
        base_stat_value, base_value_index = _pick_best_base_value(base_stat_key, ordered_values)
    if base_stat_value is None and base_stat_key == "attack_flat" and ordered_slash_base_values:
        base_stat_value, _ = _pick_best_base_value(base_stat_key, ordered_slash_base_values)
        if base_stat_value is not None:
            base_value_index = next(
                (
                    index
                    for index, (token, _) in enumerate(ordered_token_values)
                    if _numeric_equal(_coerce_slash_base_value(token), base_stat_value)
                ),
                None,
            )
    if base_stat_value is None and len(ordered_values) >= 2:
        base_stat_value = ordered_values[0]
        base_value_index = 0
    if base_stat_key is None and base_stat_value is not None:
        # W-Engine base stat is ATK in every reviewed sample and runtime crop.
        base_stat_key = "attack_flat"

    remaining_effect_token_values = [
        (token, value)
        for token, value in effect_token_values
        if not _numeric_equal(value, base_stat_value)
    ]
    advanced_stat_value = _pick_first_compatible_token_value(advanced_stat_key, advanced_line_token_values)
    if advanced_stat_value is None:
        advanced_stat_value = _pick_first_compatible_token_value(advanced_stat_key, advanced_segment_token_values)
    if advanced_stat_value is None:
        advanced_stat_value = _pick_first_compatible_token_value(advanced_stat_key, remaining_effect_token_values)
    if advanced_stat_value is None and base_value_index is not None:
        advanced_stat_value = _pick_first_compatible_token_value(
            advanced_stat_key,
            ordered_token_values[base_value_index + 1 :],
        )
    if advanced_stat_value is None:
        advanced_stat_value = _pick_first_compatible_token_value(advanced_stat_key, ordered_token_values)
    if advanced_stat_value is None and advanced_stat_key is None:
        fallback_values = (
            ordered_values[base_value_index + 1 :]
            if base_value_index is not None and base_value_index + 1 < len(ordered_values)
            else ordered_values
        )
        advanced_stat_value = _pick_first_distinct_value(
            fallback_values,
            excluding=base_stat_value,
        )

    base_stat_value = _normalize_stat_value_for_key(base_stat_key, base_stat_value)
    advanced_stat_value = _normalize_stat_value_for_key(advanced_stat_key, advanced_stat_value)
    if advanced_stat_key is None and _numeric_equal(advanced_stat_value, base_stat_value):
        advanced_stat_value = None

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
