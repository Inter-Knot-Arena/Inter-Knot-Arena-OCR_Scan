from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amplifier_identity import (
    crop_title_image,
    extract_amplifier_title_candidate,
    language_tag_for_locale,
    normalize_alias_key,
    run_winrt_ocr_batch,
)
from equipment_taxonomy import display_name_for_id, normalize_name_and_id
from manifest_lib import ensure_manifest_defaults, load_manifest


def _text(value: Any) -> str:
    return str(value or "").strip()


def _iter_reviewed_amplifiers(records: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str, Path]]:
    output: List[Tuple[Dict[str, Any], str, Path]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if _text(record.get("screenRole")) != "amplifier_detail":
            continue
        if _text(record.get("qaStatus")).lower() != "reviewed":
            continue
        labels = record.get("labels")
        if not isinstance(labels, dict) or not isinstance(labels.get("reviewFinal"), dict):
            continue
        amplifier_id = _text(labels.get("amplifier_id"))
        if not amplifier_id:
            continue
        path = Path(_text(record.get("path")))
        if not path.exists():
            continue
        output.append((record, amplifier_id, path))
    return output


def _looks_suspicious_amplifier_id(value: str) -> bool:
    token = _text(value)
    if not token:
        return True
    if token.endswith("_o"):
        return True
    if token.startswith("amp_o_"):
        return True
    if token in {"amp_i", "amp_ii", "amp_iii", "amp_iv", "amp_v", "amp_vi", "amp_mew", "amp_new"}:
        return True
    return False


def _looks_suspicious_display_name(value: str) -> bool:
    token = _text(value).lower()
    if not token:
        return True
    if token in {"i", "ii", "iii", "iv", "v", "vi"}:
        return True
    if token.endswith(" o"):
        return True
    return False


def _resolve_alias_target(title_candidate: str, review_amplifier_id: str, review_display_name: str) -> Tuple[str, str]:
    try:
        canonical_id, canonical_name = normalize_name_and_id(title_candidate, "", kind="amplifier")
        if canonical_id:
            return canonical_id, canonical_name
    except ValueError:
        pass
    fallback_name = display_name_for_id(review_amplifier_id) or review_display_name
    return review_amplifier_id, fallback_name


def _quality_score(amplifier_id: str, display_name: str) -> int:
    score = 0
    if not _looks_suspicious_amplifier_id(amplifier_id):
        score += 100
    if not _looks_suspicious_display_name(display_name):
        score += 20
    score += min(len(amplifier_id), 40)
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reviewed amplifier title alias contract from account-import OCR corpus.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output", default="contracts/amplifier-title-aliases.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    reviewed = _iter_reviewed_amplifiers(records)
    alias_buckets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    conflicts: List[Dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="amp_alias_build_") as raw_tmp:
        temp_root = Path(raw_tmp)
        crop_dir = temp_root / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        batches: Dict[str, List[Dict[str, str]]] = {}
        lookup: Dict[str, Tuple[Dict[str, Any], str]] = {}

        for record, amplifier_id, path in reviewed:
            crop_id = _text(record.get("id"))
            crop_path = crop_dir / f"{crop_id}.png"
            crop_title_image(path, crop_path)
            language_tag = language_tag_for_locale(_text(record.get("locale")))
            batches.setdefault(language_tag, []).append({"id": crop_id, "path": str(crop_path)})
            lookup[crop_id] = (record, amplifier_id)

        ocr_results: Dict[str, str] = {}
        for language_tag, crops in batches.items():
            ocr_results.update(run_winrt_ocr_batch(crops, language_tag=language_tag, temp_root=temp_root))

    for record_id, (record, amplifier_id) in lookup.items():
        raw_text = _text(ocr_results.get(record_id))
        title_candidate = extract_amplifier_title_candidate(raw_text)
        normalized_alias = normalize_alias_key(title_candidate)
        if not normalized_alias:
            continue
        review_display_name = _text((record.get("labels") or {}).get("amplifier_name"))
        canonical_id, canonical_display_name = _resolve_alias_target(title_candidate, amplifier_id, review_display_name)
        candidate_payload = {
            "alias": title_candidate,
            "normalizedAlias": normalized_alias,
            "amplifierId": canonical_id,
            "displayName": canonical_display_name,
            "reviewAmplifierId": amplifier_id,
            "reviewDisplayName": review_display_name,
            "qualityScore": _quality_score(canonical_id, canonical_display_name),
            "sampleCount": 1,
        }
        bucket = alias_buckets.setdefault(normalized_alias, {})
        existing = bucket.get(canonical_id)
        if existing is None:
            bucket[canonical_id] = candidate_payload
        else:
            existing["sampleCount"] = int(existing.get("sampleCount") or 0) + 1

    alias_map: Dict[str, Dict[str, str]] = {}
    for normalized_alias, bucket in alias_buckets.items():
        ranked = sorted(
            bucket.values(),
            key=lambda item: (
                -int(item.get("qualityScore") or 0),
                -int(item.get("sampleCount") or 0),
                str(item.get("amplifierId") or ""),
            ),
        )
        winner = ranked[0]
        alias_map[normalized_alias] = {
            "alias": _text(winner.get("alias")),
            "normalizedAlias": normalized_alias,
            "amplifierId": _text(winner.get("amplifierId")),
            "displayName": _text(winner.get("displayName")),
        }
        if len(ranked) > 1:
            for loser in ranked[1:]:
                conflicts.append(
                    {
                        "normalizedAlias": normalized_alias,
                        "existingAmplifierId": _text(winner.get("amplifierId")),
                        "newAmplifierId": _text(loser.get("amplifierId")),
                        "existingAlias": _text(winner.get("alias")),
                        "newAlias": _text(loser.get("alias")),
                    }
                )

    payload = {
        "generatedFromManifest": str(manifest_path),
        "reviewedAmplifierCount": len(reviewed),
        "aliasCount": len(alias_map),
        "conflictCount": len(conflicts),
        "entries": sorted(alias_map.values(), key=lambda item: (str(item["amplifierId"]), str(item["normalizedAlias"]))),
        "conflicts": conflicts,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
