from __future__ import annotations

import re
from pathlib import Path


SUPPORTED_LOCALES = {"RU", "EN", "UNKNOWN"}

_LOCALE_HINTS = {
    "RU": {
        "ru",
        "ru-ru",
        "russian",
        "russia",
        "rus",
        "рус",
        "русский",
        "ру",
    },
    "EN": {
        "en",
        "en-us",
        "en-gb",
        "english",
        "eng",
        "англ",
        "английский",
    },
}


def normalize_dataset_locale(value: str | None, *, default: str = "UNKNOWN") -> str:
    token = str(value or "").strip().upper()
    if token in {"MIXED", "AUTO"}:
        return token
    if token in {"UNKNOWN", ""}:
        return "UNKNOWN" if default == "UNKNOWN" else normalize_dataset_locale(default, default="UNKNOWN")
    if token in SUPPORTED_LOCALES:
        return token
    return normalize_dataset_locale(default, default="UNKNOWN")


def _path_tokens(path: Path, root: Path) -> list[str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    joined = "/".join(relative.parts)
    raw_tokens = re.split(r"[^0-9A-Za-z\u0400-\u04FF]+", joined.lower())
    return [token for token in raw_tokens if token]


def detect_locale_from_path(path: Path, root: Path) -> str:
    tokens = set(_path_tokens(path, root))
    for locale, hints in _LOCALE_HINTS.items():
        if tokens.intersection(hints):
            return locale
    return "UNKNOWN"


def resolve_record_locale(path: Path, root: Path, requested_locale: str, *, fallback: str = "UNKNOWN") -> str:
    normalized_request = normalize_dataset_locale(requested_locale, default=fallback)
    if normalized_request in {"RU", "EN"}:
        return normalized_request
    if normalized_request in {"AUTO", "MIXED"}:
        detected = detect_locale_from_path(path, root)
        if detected in {"RU", "EN"}:
            return detected
        return normalize_dataset_locale(fallback, default="UNKNOWN")
    return normalize_dataset_locale(fallback, default="UNKNOWN")
