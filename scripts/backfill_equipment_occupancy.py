from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import (
    ACCOUNT_IMPORT_WORKFLOW,
    EQUIPMENT_ROLE,
    filter_records,
    record_screen_role,
    source_index_from_manifest,
)


def _labels(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = record.get("labels")
    if not isinstance(payload, dict):
        payload = {}
        record["labels"] = payload
    return payload


def _slot_occupancy(summary: List[Dict[str, Any]]) -> Dict[str, bool]:
    occupied = {
        str(int(item["slot"]))
        for item in summary
        if isinstance(item, dict) and isinstance(item.get("slot"), int)
    }
    return {str(slot): str(slot) in occupied for slot in range(1, 7)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill equipment occupancy fields from reviewed equipmentOverview labels.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    source_index = source_index_from_manifest(manifest.get("sources", []))
    records = filter_records(
        manifest.get("records", []),
        source_index=source_index,
        workflow=ACCOUNT_IMPORT_WORKFLOW,
        import_eligible_only=True,
    )

    updated = 0
    for record in records:
        if record_screen_role(record, source_index) != EQUIPMENT_ROLE:
            continue
        if str(record.get("qaStatus") or "").lower() != "reviewed":
            continue
        labels = _labels(record)
        overview = labels.get("equipmentOverview")
        if not isinstance(overview, dict):
            continue

        summary = labels.get("equipment_disc_summary")
        if not isinstance(summary, list):
            summary = overview.get("discs")
        if not isinstance(summary, list):
            summary = []

        weapon = overview.get("weapon")
        weapon_present = bool(isinstance(weapon, dict) and weapon)
        slot_occupancy = _slot_occupancy(summary)

        changed = False
        if labels.get("weapon_present") != weapon_present:
            labels["weapon_present"] = weapon_present
            changed = True
        if labels.get("disc_slot_occupancy") != slot_occupancy:
            labels["disc_slot_occupancy"] = slot_occupancy
            changed = True

        if overview.get("weaponPresent") != weapon_present:
            overview["weaponPresent"] = weapon_present
            changed = True
        if overview.get("discSlotOccupancy") != slot_occupancy:
            overview["discSlotOccupancy"] = slot_occupancy
            changed = True

        if changed:
            updated += 1

    if updated:
        qa_status = manifest.setdefault("qaStatus", {})
        if isinstance(qa_status, dict):
            qa_status["qaUpdatedAt"] = utc_now()
        save_manifest(manifest_path, manifest)

    print(json.dumps({"updatedRecords": updated}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
