from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "contracts" / "equipment-taxonomy.json"
DEFAULT_PREVIEW_PATH = REPO_ROOT / "docs" / "dataset_preview.csv"
DEFAULT_WEB_ROOT = REPO_ROOT.parent / "Inter-Knot Arena web"
DEFAULT_DISC_CATALOG_PATH = DEFAULT_WEB_ROOT / "packages" / "catalog" / "disc-sets.v1.json"


def _normalize_text(value: str) -> str:
    token = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    token = token.replace("\u2013", "-")
    token = token.replace("\u2014", "-")
    token = token.replace("\u2212", "-")
    token = token.replace("\u00a0", " ")
    token = token.replace("&", " and ")
    token = token.replace("/", " ")
    token = token.replace("_", " ")
    token = token.replace("w-engine", "w engine")
    token = token.replace("w engine", "wengine")
    token = re.sub(r"\s*\[[1-6]\]\s*$", "", token)
    token = re.sub(r"\s*\((?:slot|disc|disk)\s*[1-6]\)\s*$", "", token, flags=re.IGNORECASE)
    token = re.sub(r"[^0-9a-zA-Z\u0400-\u04FF+\- ]+", " ", token)
    return " ".join(token.split())


def _slugify(value: str) -> str:
    token = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    token = token.replace("&", " and ")
    token = token.replace("/", " ")
    token = token.replace("-", " ")
    token = re.sub(r"[^0-9a-zA-Z]+", " ", token)
    return "_".join(part for part in token.split() if part)


def _humanize_legacy_id(identifier: str, prefix: str) -> str:
    token = str(identifier or "").strip()
    if not token.startswith(prefix):
        return ""
    body = token[len(prefix) :]
    if not body:
        return ""
    return " ".join(part.capitalize() for part in body.split("_") if part)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _iter_reviewed_preview_rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader if isinstance(row, dict)]
    return rows


def _looks_corrupted(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return True
    meaningful = [char for char in token if not char.isspace() and char not in "-_[]()"]
    if not meaningful:
        return True
    question_count = sum(1 for char in meaningful if char == "?")
    mojibake_markers = ("Р", "Ѓ", "Є", "ё", "ї", "ј")
    marker_hits = sum(1 for char in meaningful if char in mojibake_markers)
    if marker_hits and (marker_hits / len(meaningful)) >= 0.35:
        return True
    return question_count == len(meaningful)


def _pick_display_name(preferred: str, fallback: str) -> str:
    preferred_text = str(preferred or "").strip()
    if preferred_text and not _looks_corrupted(preferred_text):
        return preferred_text
    fallback_text = str(fallback or "").strip()
    if fallback_text and not _looks_corrupted(fallback_text):
        return fallback_text
    return fallback_text or preferred_text


def _append_alias(bucket: list[str], value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    if text not in bucket:
        bucket.append(text)


def _build_disc_entries(preview_rows: Iterable[dict[str, str]], disc_catalog_path: Path) -> list[dict[str, Any]]:
    preview_by_id: dict[str, dict[str, Any]] = {}
    preview_names: defaultdict[str, list[str]] = defaultdict(list)
    for row in preview_rows:
        canonical_id = str(row.get("reviewed_disc_set_id") or "").strip()
        display_name = str(row.get("reviewed_disc_set_name") or "").strip()
        if not canonical_id and not display_name:
            continue
        if canonical_id:
            entry = preview_by_id.setdefault(
                canonical_id,
                {
                    "discSetId": canonical_id,
                    "displayName": "",
                    "aliases": [],
                },
            )
            if display_name and not entry["displayName"]:
                entry["displayName"] = display_name
            _append_alias(entry["aliases"], canonical_id)
            _append_alias(entry["aliases"], display_name)
        if display_name:
            preview_names[_normalize_text(display_name)].append(canonical_id)

    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if disc_catalog_path.exists():
        payload = _load_json(disc_catalog_path)
        for raw in payload.get("discSets", []):
            if not isinstance(raw, dict):
                continue
            english_name = str(raw.get("name") or "").strip()
            slug = _slugify(english_name)
            if not slug:
                continue
            legacy_id = f"set_{slug}"
            modern_id = f"discset_{slug}"
            preview_entry = preview_by_id.get(legacy_id)
            preview_match_ids = [candidate for candidate in preview_names.get(_normalize_text(english_name), []) if candidate]
            canonical_id = legacy_id
            if preview_entry is None and preview_match_ids:
                canonical_id = preview_match_ids[0]
                preview_entry = preview_by_id.get(canonical_id)
            display_name = _pick_display_name(english_name, preview_entry["displayName"] if isinstance(preview_entry, dict) else "")
            aliases: list[str] = []
            for value in (
                canonical_id,
                legacy_id,
                modern_id,
                str(raw.get("discSetId") or ""),
                str(raw.get("gameId") or ""),
                english_name,
            ):
                _append_alias(aliases, value)
            if isinstance(preview_entry, dict):
                for alias in preview_entry.get("aliases", []):
                    _append_alias(aliases, str(alias))
            entries.append(
                {
                    "discSetId": canonical_id,
                    "displayName": display_name,
                    "aliases": aliases,
                    "source": {
                        "catalog": "web.disc-sets.v1",
                        "gameId": raw.get("gameId"),
                    },
                }
            )
            seen_ids.add(canonical_id)

    for canonical_id, preview_entry in sorted(preview_by_id.items()):
        if canonical_id in seen_ids:
            continue
        display_name = _pick_display_name(preview_entry.get("displayName", ""), "")
        aliases: list[str] = []
        _append_alias(aliases, canonical_id)
        for alias in preview_entry.get("aliases", []):
            _append_alias(aliases, str(alias))
        entries.append(
            {
                "discSetId": canonical_id,
                "displayName": display_name,
                "aliases": aliases,
                "source": {
                    "catalog": "dataset_preview",
                },
            }
        )
    entries.sort(key=lambda item: str(item["discSetId"]))
    return entries


def _build_amplifier_entries(preview_rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in preview_rows:
        canonical_id = str(row.get("reviewed_amplifier_id") or "").strip()
        display_name = str(row.get("reviewed_amplifier_name") or "").strip()
        if not canonical_id and not display_name:
            continue
        if not canonical_id and display_name:
            slug = _slugify(display_name)
            if slug:
                canonical_id = f"amp_{slug}"
        if not canonical_id:
            continue
        entry = by_id.setdefault(
            canonical_id,
            {
                "amplifierId": canonical_id,
                "displayName": "",
                "aliases": [],
            },
        )
        if display_name and not entry["displayName"]:
            entry["displayName"] = display_name
        _append_alias(entry["aliases"], canonical_id)
        _append_alias(entry["aliases"], display_name)
        if display_name:
            slug = _slugify(display_name)
            if slug:
                _append_alias(entry["aliases"], f"weapon_{slug}")
                _append_alias(entry["aliases"], slug)

    entries: list[dict[str, Any]] = []
    for canonical_id, entry in sorted(by_id.items()):
        fallback_english_name = _humanize_legacy_id(canonical_id, "amp_")
        display_name = _pick_display_name(fallback_english_name, entry.get("displayName", ""))
        aliases: list[str] = []
        _append_alias(aliases, canonical_id)
        for alias in entry.get("aliases", []):
            _append_alias(aliases, str(alias))
        if fallback_english_name:
            _append_alias(aliases, fallback_english_name)
            _append_alias(aliases, f"weapon_{_slugify(fallback_english_name)}")
        if not display_name:
            display_name = fallback_english_name
        entries.append(
            {
                "amplifierId": canonical_id,
                "displayName": display_name,
                "aliases": aliases,
                "source": {
                    "catalog": "dataset_preview",
                },
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync OCR equipment taxonomy snapshot from local catalogs and reviewed preview data.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--preview-csv", default=str(DEFAULT_PREVIEW_PATH))
    parser.add_argument("--disc-catalog", default=str(DEFAULT_DISC_CATALOG_PATH))
    args = parser.parse_args()

    preview_rows = list(_iter_reviewed_preview_rows(Path(args.preview_csv).resolve()))
    disc_entries = _build_disc_entries(preview_rows, Path(args.disc_catalog).resolve())
    amplifier_entries = _build_amplifier_entries(preview_rows)

    payload = {
        "version": "2026-03-12",
        "sources": {
            "datasetPreview": str(Path(args.preview_csv).resolve()),
            "discCatalog": str(Path(args.disc_catalog).resolve()),
        },
        "discSets": disc_entries,
        "amplifiers": amplifier_entries,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(
        json.dumps(
            {
                "output": str(output_path),
                "discSetCount": len(disc_entries),
                "amplifierCount": len(amplifier_entries),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
