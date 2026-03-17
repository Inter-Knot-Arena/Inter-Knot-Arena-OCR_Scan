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

EN_LANGUAGE_TAG = "en-US"
RU_LANGUAGE_TAG = "ru"


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
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"(?i)\b(?:lvl?|lv\.?|ур\.?)\s*[0-9oiIl/]+\b", " ", normalized)
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


def crop_title_image(source_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to decode image: {source_path}")
    height, width = image.shape[:2]
    x0 = int(width * TITLE_CROP_BOX[0])
    y0 = int(height * TITLE_CROP_BOX[1])
    x1 = int(width * TITLE_CROP_BOX[2])
    y1 = int(height * TITLE_CROP_BOX[3])
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Empty title crop: {source_path}")
    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), crop):
        raise ValueError(f"Failed to write title crop: {output_path}")


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
