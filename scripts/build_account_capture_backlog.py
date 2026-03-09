from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from manifest_lib import ensure_manifest_defaults, load_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import (
    ACCOUNT_IMPORT_WORKFLOW,
    AMPLIFIER_DETAIL_ROLE,
    DISK_DETAIL_ROLE,
    EQUIPMENT_ROLE,
    MINDSCAPE_ROLE,
    filter_records,
    record_head,
    record_screen_role,
    source_index_from_manifest,
)
from roster_taxonomy import canonicalize_agent_label, current_agent_ids


def _labels(record: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = record.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _agent_icon_label(record: Dict[str, Any]) -> str:
    for payload_key in ("labels", "suggestedLabels"):
        payload = _labels(record, payload_key)
        for key in ("agent_icon_id", "agentId", "label"):
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            canonical = canonicalize_agent_label(value.strip())
            if canonical and canonical != "unknown":
                return canonical
    return ""


def _owned_agent_ids(record: Dict[str, Any]) -> List[str]:
    labels = _labels(record, "labels")
    values = labels.get("owned_agent_ids")
    if not isinstance(values, list):
        return []
    output: List[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            output.append(value.strip())
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build account-import capture backlog for OCR.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-file", default="docs/account_capture_backlog.json")
    parser.add_argument("--target-uid-panel", type=int, default=80)
    parser.add_argument("--target-roster-per-agent", type=int, default=8)
    parser.add_argument("--target-agent-detail-per-agent", type=int, default=10)
    parser.add_argument("--target-equipment-frames", type=int, default=300)
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = source_index_from_manifest(manifest.get("sources", []))
    scoped_records = filter_records(records, source_index, workflow=ACCOUNT_IMPORT_WORKFLOW, import_eligible_only=True)

    role_head_counts: Counter[str] = Counter()
    roster_counts: Counter[str] = Counter()
    detail_counts: Counter[str] = Counter()
    equipment_frames = 0
    disk_detail_frames = 0
    amplifier_detail_frames = 0
    optional_mindscape_frames = 0
    agent_detail_mindscape_frames = 0
    roster_owned_agents: set[str] = set()
    roster_not_owned_agents: set[str] = set()

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        head = record_head(record)
        role = record_screen_role(record, source_index)
        role_head_counts[f"{role}:{head}"] += 1
        if head == "agent_icon":
            agent_id = _agent_icon_label(record)
            if role == "roster":
                owned = _owned_agent_ids(record)
                if owned:
                    roster_owned_agents.update(owned)
                    not_owned_values = _labels(record, "labels").get("not_owned_agent_ids")
                    if isinstance(not_owned_values, list):
                        roster_not_owned_agents.update(
                            str(item).strip() for item in not_owned_values if isinstance(item, str) and str(item).strip()
                        )
            if not agent_id:
                continue
            if role == "roster":
                roster_counts[agent_id] += 1
            elif role == "agent_detail":
                detail_counts[agent_id] += 1
        elif head == "equipment":
            if role == EQUIPMENT_ROLE:
                equipment_frames += 1
            elif role == DISK_DETAIL_ROLE:
                disk_detail_frames += 1
            elif role == AMPLIFIER_DETAIL_ROLE:
                amplifier_detail_frames += 1
            elif role == MINDSCAPE_ROLE:
                optional_mindscape_frames += 1
        if role == "agent_detail" and _labels(record, "labels").get("agent_mindscape") is not None:
            agent_detail_mindscape_frames += 1

    roster_agents = current_agent_ids()
    roster_deficits = {agent: max(0, int(args.target_roster_per_agent) - roster_counts.get(agent, 0)) for agent in roster_agents}
    detail_deficits = {agent: max(0, int(args.target_agent_detail_per_agent) - detail_counts.get(agent, 0)) for agent in roster_agents}

    backlog = {
        "generatedAt": utc_now(),
        "workflow": ACCOUNT_IMPORT_WORKFLOW,
        "eligibleRecordCount": len(scoped_records),
        "roleHeadCounts": dict(role_head_counts),
        "targets": {
            "uidPanelFrames": int(args.target_uid_panel),
            "rosterOwnershipAgents": len(roster_agents),
            "rosterPerAgent": int(args.target_roster_per_agent),
            "agentDetailPerAgent": int(args.target_agent_detail_per_agent),
            "equipmentFrames": int(args.target_equipment_frames),
        },
        "current": {
            "uidPanelFrames": int(role_head_counts.get("uid_panel:uid_digit", 0)),
            "rosterCoveredAgents": len(roster_owned_agents | roster_not_owned_agents),
            "rosterIconLabeledAgents": sum(1 for agent in roster_agents if roster_counts.get(agent, 0) > 0),
            "rosterOwnedAgents": len(roster_owned_agents),
            "rosterNotOwnedAgents": len(roster_not_owned_agents),
            "agentDetailCoveredAgents": sum(1 for agent in roster_agents if detail_counts.get(agent, 0) > 0),
            "equipmentFrames": int(equipment_frames),
            "diskDetailFrames": int(disk_detail_frames),
            "amplifierDetailFrames": int(amplifier_detail_frames),
            "agentDetailMindscapeFrames": int(agent_detail_mindscape_frames),
            "optionalMindscapeFrames": int(optional_mindscape_frames),
        },
        "uidPanelMissingFrames": max(0, int(args.target_uid_panel) - int(role_head_counts.get("uid_panel:uid_digit", 0))),
        "missingRosterAgents": [agent for agent in roster_agents if agent not in roster_owned_agents and agent not in roster_not_owned_agents],
        "missingRosterIconAgents": [agent for agent, deficit in roster_deficits.items() if deficit > 0],
        "missingAgentDetailAgents": [agent for agent, deficit in detail_deficits.items() if deficit > 0],
        "ownedAgentIds": [agent for agent in roster_agents if agent in roster_owned_agents],
        "notOwnedAgentIds": [agent for agent in roster_agents if agent in roster_not_owned_agents],
        "equipmentMissingFrames": max(0, int(args.target_equipment_frames) - equipment_frames),
        "rosterDeficits": roster_deficits,
        "agentDetailDeficits": detail_deficits,
        "recommendedCaptureOrder": [
            "uid_panel",
            "roster",
            "agent_detail",
            "equipment",
            "disk_detail",
            "amplifier_detail",
        ],
        "recommendedCommands": [
            "python scripts/session_capture.py --manifest dataset_manifest.json --head uid_digit --workflow account_import --screen-role uid_panel --duration-sec 60 --fps 1.0 --locale RU --resolution 1080p",
            "python scripts/session_capture.py --manifest dataset_manifest.json --head agent_icon --workflow account_import --screen-role roster --duration-sec 120 --fps 1.0 --locale RU --resolution 1080p",
            "python scripts/session_capture.py --manifest dataset_manifest.json --head agent_icon --workflow account_import --screen-role agent_detail --duration-sec 120 --fps 1.0 --locale RU --resolution 1080p",
            "python scripts/session_capture.py --manifest dataset_manifest.json --head equipment --workflow account_import --screen-role equipment --duration-sec 180 --fps 0.5 --locale RU --resolution 1080p",
            "python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\\IKA_DATA\\ocr\\drops\\batch_001 --locale RU",
        ],
    }

    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(backlog, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(backlog, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
