from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amplifier_identity import (  # noqa: E402
    classify_amplifier_text,
    confidence_for_source_mode,
    crop_title_image,
    language_tag_for_locale,
    run_winrt_ocr_batch,
)
from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest  # noqa: E402


def _text(value: Any) -> str:
    return str(value or "").strip()


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


def _iter_reviewed_amplifier_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
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
        path = Path(_text(record.get("path")))
        if not path.exists():
            continue
        if not _text(labels.get("amplifier_id")):
            continue
        output.append(record)
    return output


def _update_record_labels(record: Dict[str, Any], predicted_weapon_id: str, predicted_display_name: str, source_mode: str) -> Dict[str, str]:
    labels = record.setdefault("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("record.labels must be an object")

    old_id = _text(labels.get("amplifier_id"))
    old_name = _text(labels.get("amplifier_name"))
    labels["amplifier_id"] = predicted_weapon_id
    labels["amplifier_name"] = predicted_display_name
    labels["label"] = predicted_weapon_id

    weapon = labels.get("weapon")
    if not isinstance(weapon, dict):
        weapon = {}
    weapon["weaponId"] = predicted_weapon_id
    weapon["displayName"] = predicted_display_name
    weapon["weaponPresent"] = bool(labels.get("weapon_present", True))
    agent_id = _text(labels.get("equipment_agent_id"))
    if agent_id:
        weapon["agentId"] = agent_id
    labels["weapon"] = weapon

    review_final = labels.get("reviewFinal")
    if not isinstance(review_final, dict):
        review_final = {}
        labels["reviewFinal"] = review_final
    review_final["reviewedAt"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    note = f"Canonicalized amplifier id from {old_id or '<missing>'} to {predicted_weapon_id} via {source_mode}."
    existing_notes = _text(review_final.get("notes"))
    review_final["notes"] = f"{existing_notes} {note}".strip() if existing_notes else note

    return {
        "oldAmplifierId": old_id,
        "newAmplifierId": predicted_weapon_id,
        "oldAmplifierName": old_name,
        "newAmplifierName": predicted_display_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize reviewed amplifier_detail labels using the reviewed title alias contract.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--report", default="docs/canonicalize_amplifier_review_ids.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    reviewed_records = _iter_reviewed_amplifier_records(records)
    ocr_results: Dict[str, str] = {}
    lookup: Dict[str, Dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="amp_canonicalize_") as raw_tmp:
        temp_root = Path(raw_tmp)
        crop_dir = temp_root / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        batches: Dict[str, List[Dict[str, str]]] = {}

        for record in reviewed_records:
            record_id = _text(record.get("id"))
            source_path = Path(_text(record.get("path")))
            crop_path = crop_dir / f"{record_id}.png"
            crop_title_image(source_path, crop_path)
            language_tag = language_tag_for_locale(_text(record.get("locale")))
            batches.setdefault(language_tag, []).append({"id": record_id, "path": str(crop_path)})
            lookup[record_id] = record

        for language_tag, crops in batches.items():
            ocr_results.update(run_winrt_ocr_batch(crops, language_tag=language_tag, temp_root=temp_root))

    updated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for record_id, record in lookup.items():
        labels = record.get("labels")
        if not isinstance(labels, dict):
            continue
        current_id = _text(labels.get("amplifier_id"))
        current_name = _text(labels.get("amplifier_name"))
        prediction = classify_amplifier_text(_text(ocr_results.get(record_id)))
        if prediction is None:
            skipped.append({"recordId": record_id, "reason": "no_prediction", "currentAmplifierId": current_id})
            continue
        if prediction.source_mode not in {"alias_contract", "alias_contract_fuzzy"}:
            skipped.append({"recordId": record_id, "reason": prediction.source_mode, "currentAmplifierId": current_id})
            continue
        should_update = current_id != prediction.weapon_id and (
            _looks_suspicious_amplifier_id(current_id)
            or confidence_for_source_mode(prediction.source_mode) >= 0.96
        )
        if not should_update:
            continue
        delta = _update_record_labels(record, prediction.weapon_id, prediction.display_name, prediction.source_mode)
        updated.append(
            {
                "recordId": record_id,
                "path": _text(record.get("path")),
                "sourceMode": prediction.source_mode,
                "rawText": prediction.raw_text,
                "titleCandidate": prediction.title_candidate,
                **delta,
            }
        )

    save_manifest(manifest_path, manifest)

    report = {
        "manifest": str(manifest_path),
        "reviewedAmplifierRecordCount": len(reviewed_records),
        "updatedCount": len(updated),
        "skippedCount": len(skipped),
        "updated": updated,
        "skipped": skipped[:50],
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
