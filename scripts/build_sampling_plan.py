from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from manifest_lib import ensure_manifest_defaults, load_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, filter_records, source_index_from_manifest
from roster_taxonomy import canonicalize_agent_label, current_agent_ids


def _label_payload(record: Dict[str, Any], suggested: bool) -> Dict[str, Any]:
    key = "suggestedLabels" if suggested else "labels"
    labels = record.get(key)
    if isinstance(labels, dict):
        return labels
    return {}


def _extract_head(record: Dict[str, Any], *, suggested: bool) -> str:
    labels = _label_payload(record, suggested=suggested)
    if labels:
        head = labels.get("head")
        if isinstance(head, str) and head.strip():
            return head.strip()
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def _extract_label(record: Dict[str, Any], *, suggested: bool) -> str:
    labels = _label_payload(record, suggested=suggested)
    if labels:
        for key in ("uid_digit", "agent_icon_id", "amplifier_id", "disc_set_id", "label"):
            value = labels.get(key)
            if isinstance(value, str) and value.strip():
                raw = value.strip()
                if _extract_head(record, suggested=suggested) == "agent_icon":
                    return canonicalize_agent_label(raw) or "unknown"
                return raw
    for key in ("label", "agentId"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            if _extract_head(record, suggested=suggested) == "agent_icon":
                return canonicalize_agent_label(raw) or "unknown"
            return raw
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build class-balance sampling plan from OCR dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--target-uid-digit", type=int, default=20000)
    parser.add_argument("--target-agent-icon", type=int, default=15000)
    parser.add_argument("--target-equipment", type=int, default=15000)
    parser.add_argument("--output-file", default="")
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true", default=True)
    args = parser.parse_args()

    manifest = ensure_manifest_defaults(load_manifest(Path(args.manifest).resolve()))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    source_index = source_index_from_manifest(manifest.get("sources", []))
    scoped_records = filter_records(
        records,
        source_index=source_index,
        workflow=args.workflow,
        import_eligible_only=bool(args.import_eligible_only),
    )

    reviewed_head_counts: Counter[str] = Counter()
    suggested_head_counts: Counter[str] = Counter()
    reviewed_label_counts: Counter[str] = Counter()
    suggested_label_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        reviewed_head = _extract_head(record, suggested=False)
        suggested_head = _extract_head(record, suggested=True)
        reviewed_label = _extract_label(record, suggested=False)
        suggested_label = _extract_label(record, suggested=True)
        if reviewed_head:
            reviewed_head_counts[reviewed_head] += 1
        if suggested_head:
            suggested_head_counts[suggested_head] += 1
        if reviewed_label:
            reviewed_label_counts[reviewed_label] += 1
        if suggested_label:
            suggested_label_counts[suggested_label] += 1
        locale_counts[str(record.get("locale") or "unknown")] += 1
        resolution_counts[str(record.get("resolution") or "unknown")] += 1
        source_counts[str(record.get("sourceId") or "src_unknown")] += 1

    deficits = {
        "uid_digit": max(0, max(1, args.target_uid_digit) - reviewed_head_counts.get("uid_digit", 0)),
        "agent_icon": max(0, max(1, args.target_agent_icon) - reviewed_head_counts.get("agent_icon", 0)),
        "equipment": max(0, max(1, args.target_equipment) - reviewed_head_counts.get("equipment", 0)),
    }
    agent_icon_counts = {
        agent: int(reviewed_label_counts.get(agent, 0))
        for agent in current_agent_ids()
    }
    suggested_agent_icon_counts = {
        agent: int(suggested_label_counts.get(agent, 0))
        for agent in current_agent_ids()
    }
    missing_agent_icons = [agent for agent, count in agent_icon_counts.items() if count <= 0]

    per_locale_resolution = defaultdict(int)
    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        key = f"{str(record.get('locale') or 'unknown')}:{str(record.get('resolution') or 'unknown')}"
        per_locale_resolution[key] += 1

    plan = {
        "recordCount": len(scoped_records),
        "workflow": str(args.workflow),
        "importEligibleOnly": bool(args.import_eligible_only),
        "headCounts": dict(reviewed_head_counts),
        "reviewedHeadCounts": dict(reviewed_head_counts),
        "suggestedHeadCounts": dict(suggested_head_counts),
        "labelCounts": dict(reviewed_label_counts),
        "reviewedLabelCounts": dict(reviewed_label_counts),
        "suggestedLabelCounts": dict(suggested_label_counts),
        "agentIconCounts": agent_icon_counts,
        "suggestedAgentIconCounts": suggested_agent_icon_counts,
        "rosterAgentCount": len(current_agent_ids()),
        "coveredAgentCount": sum(1 for count in agent_icon_counts.values() if count > 0),
        "suggestedCoveredAgentCount": sum(1 for count in suggested_agent_icon_counts.values() if count > 0),
        "missingAgentIcons": missing_agent_icons,
        "missingSuggestedAgentIcons": [agent for agent, count in suggested_agent_icon_counts.items() if count <= 0],
        "localeCounts": dict(locale_counts),
        "resolutionCounts": dict(resolution_counts),
        "sourceCounts": dict(source_counts),
        "localeResolutionCounts": dict(per_locale_resolution),
        "deficits": deficits,
    }

    if args.output_file:
        output_path = Path(args.output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(plan, fh, ensure_ascii=True, indent=2)
            fh.write("\n")

    print(json.dumps(plan, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
