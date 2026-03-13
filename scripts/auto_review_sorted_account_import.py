from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import canonicalize_agent_label


ACCOUNT_IMPORT_WORKFLOW = "account_import"
AGENT_DETAIL_ROLE = "agent_detail"
AMPLIFIER_DETAIL_ROLE = "amplifier_detail"
DISK_DETAIL_ROLE = "disk_detail"
EQUIPMENT_ROLE = "equipment"

_ORDERED_FILE_PATTERN = re.compile(r"^(?P<prefix>\d+)_(?P<index>\d+)$")


def _record_source_index(sources: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(source.get("sourceId") or ""): source
        for source in sources
        if isinstance(source, dict) and str(source.get("sourceId") or "")
    }


def _reviewed_final(record: Dict[str, Any]) -> bool:
    labels = record.get("labels")
    return (
        isinstance(labels, dict)
        and str(record.get("qaStatus") or "").strip().lower() == "reviewed"
        and isinstance(labels.get("reviewFinal"), dict)
    )


def _labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if isinstance(labels, dict):
        return labels
    labels = {}
    record["labels"] = labels
    return labels


def _normalized_focus_agent(record: Dict[str, Any], source: Dict[str, Any]) -> str:
    for raw in (record.get("focusAgentId"), source.get("focusAgentId")):
        if isinstance(raw, str) and raw.strip():
            canonical = canonicalize_agent_label(raw.strip())
            if canonical:
                return canonical
    return ""


def _origin_path(source: Dict[str, Any]) -> str:
    raw = source.get("originPath")
    return str(raw).strip() if isinstance(raw, str) else ""


def _ordered_origin_key(origin_path: str) -> Optional[Tuple[str, int]]:
    if not origin_path:
        return None
    match = _ORDERED_FILE_PATTERN.match(Path(origin_path).stem)
    if not match:
        return None
    return match.group("prefix"), int(match.group("index"))


def _review_payload(*, reviewer: str, notes: str) -> Dict[str, Any]:
    return {
        "reviewer": reviewer,
        "reviewedAt": utc_now(),
        "notes": notes,
    }


def _review_agent_detail(record: Dict[str, Any], *, focus_agent_id: str, reviewer: str) -> bool:
    labels = _labels(record)
    existing = canonicalize_agent_label(labels.get("agent_icon_id") or labels.get("label") or "")
    if existing and existing != focus_agent_id:
        return False
    labels["head"] = "agent_icon"
    labels["agent_icon_id"] = focus_agent_id
    labels["label"] = focus_agent_id
    labels["reviewFinal"] = _review_payload(
        reviewer=reviewer,
        notes="Auto-reviewed from manually sorted account-import folder; focusAgentId matched the agent folder.",
    )
    record["qaStatus"] = "reviewed"
    return True


def _empty_occupancy() -> Dict[str, bool]:
    return {str(slot): False for slot in range(1, 7)}


def _review_equipment_overview(
    record: Dict[str, Any],
    *,
    focus_agent_id: str,
    occupied_slots: List[int],
    weapon_present: bool,
    reviewer: str,
) -> bool:
    labels = _labels(record)
    existing = canonicalize_agent_label(labels.get("equipment_agent_id") or labels.get("agent_icon_id") or "")
    if existing and existing != focus_agent_id:
        return False

    occupancy = _empty_occupancy()
    for slot in occupied_slots:
        occupancy[str(slot)] = True

    summary = [{"slot": slot} for slot in occupied_slots]
    labels["head"] = "equipment"
    labels["equipment_agent_id"] = focus_agent_id
    labels["equipment_disc_summary"] = summary
    labels["weapon_present"] = bool(weapon_present)
    labels["disc_slot_occupancy"] = occupancy
    labels["label"] = focus_agent_id
    labels["equipmentOverview"] = {
        "agentId": focus_agent_id,
        "weaponPresent": bool(weapon_present),
        "discSlotOccupancy": occupancy,
        "discs": summary,
    }
    labels["reviewFinal"] = _review_payload(
        reviewer=reviewer,
        notes="Auto-reviewed from manually sorted account-import pack; occupancy derived from ordered disk folders and amplifier presence.",
    )
    record["qaStatus"] = "reviewed"
    return True


def _pack_items(records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    packs: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for item in records:
        if item["screenRole"] == AGENT_DETAIL_ROLE and current:
            packs.append(current)
            current = [item]
            continue
        current.append(item)
    if current:
        packs.append(current)
    return packs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-review sorted account-import records using focusAgentId and ordered screenshot packs."
    )
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--session", action="append", default=[], help="Limit to specific sessionId values.")
    parser.add_argument("--reviewer-id", default="auto_sorted_account_import")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    source_index = _record_source_index(manifest.get("sources", []))
    session_filter = {str(item).strip() for item in args.session if str(item).strip()}

    grouped_records: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    skipped_equipment_records: List[str] = []

    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        if str(record.get("workflow") or "").strip() != ACCOUNT_IMPORT_WORKFLOW:
            continue
        session_id = str(record.get("sessionId") or "").strip()
        if session_filter and session_id not in session_filter:
            continue

        source = source_index.get(str(record.get("sourceId") or ""))
        if not isinstance(source, dict):
            continue

        role = str(record.get("screenRole") or source.get("screenRole") or "").strip()
        focus_agent_id = _normalized_focus_agent(record, source)
        if role == AGENT_DETAIL_ROLE and focus_agent_id and not _reviewed_final(record):
            if _review_agent_detail(record, focus_agent_id=focus_agent_id, reviewer=args.reviewer_id):
                stats["reviewedAgentDetail"] += 1
            else:
                stats["skippedAgentDetailConflict"] += 1

        if role not in {AGENT_DETAIL_ROLE, EQUIPMENT_ROLE, AMPLIFIER_DETAIL_ROLE, DISK_DETAIL_ROLE}:
            continue
        if not focus_agent_id:
            stats["skippedMissingFocusAgent"] += 1
            continue

        ordered_key = _ordered_origin_key(_origin_path(source))
        if ordered_key is None:
            stats["skippedMissingOrderedOrigin"] += 1
            continue

        prefix, order_index = ordered_key
        grouped_records[(session_id, focus_agent_id, prefix)].append(
            {
                "orderIndex": order_index,
                "screenRole": role,
                "record": record,
                "source": source,
            }
        )

    for group_key, items in grouped_records.items():
        stats["orderedGroups"] += 1
        ordered_items = sorted(items, key=lambda item: int(item["orderIndex"]))
        packs = _pack_items(ordered_items)
        stats["orderedPacks"] += len(packs)

        for pack in packs:
            slot_counts: Counter[int] = Counter()
            equipment_records: List[Dict[str, Any]] = []
            amplifier_records = 0
            for item in pack:
                role = str(item["screenRole"])
                if role == EQUIPMENT_ROLE:
                    equipment_records.append(item["record"])
                elif role == AMPLIFIER_DETAIL_ROLE:
                    amplifier_records += 1
                elif role == DISK_DETAIL_ROLE:
                    capture_hints = item["record"].get("captureHints")
                    if not isinstance(capture_hints, dict):
                        capture_hints = item["source"].get("captureHints")
                    slot = None
                    if isinstance(capture_hints, dict):
                        raw_slot = capture_hints.get("slotIndex")
                        if isinstance(raw_slot, (int, float)):
                            slot = int(raw_slot)
                    if slot is not None and 1 <= slot <= 6:
                        slot_counts[slot] += 1

            if not equipment_records:
                continue

            duplicate_slots = [slot for slot, count in slot_counts.items() if count > 1]
            if len(equipment_records) != 1 or amplifier_records > 1 or duplicate_slots:
                stats["skippedAmbiguousEquipmentPacks"] += 1
                stats["skippedAmbiguousEquipmentRecords"] += len(equipment_records)
                skipped_equipment_records.extend(str(record.get("id") or "") for record in equipment_records)
                continue

            occupied_slots = sorted(slot_counts)
            equipment_record = equipment_records[0]
            if _reviewed_final(equipment_record):
                continue

            focus_agent_id = _normalized_focus_agent(equipment_record, source_index.get(str(equipment_record.get("sourceId") or ""), {}))
            if not focus_agent_id:
                stats["skippedEquipmentMissingFocusAgent"] += 1
                skipped_equipment_records.append(str(equipment_record.get("id") or ""))
                continue

            if _review_equipment_overview(
                equipment_record,
                focus_agent_id=focus_agent_id,
                occupied_slots=occupied_slots,
                weapon_present=bool(amplifier_records),
                reviewer=args.reviewer_id,
            ):
                stats["reviewedEquipment"] += 1
            else:
                stats["skippedEquipmentConflict"] += 1
                skipped_equipment_records.append(str(equipment_record.get("id") or ""))

    payload = {
        "manifest": str(manifest_path),
        "reviewerId": str(args.reviewer_id),
        "sessionFilter": sorted(session_filter),
        "stats": {key: int(value) for key, value in sorted(stats.items())},
        "skippedEquipmentRecordIds": skipped_equipment_records,
    }

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
