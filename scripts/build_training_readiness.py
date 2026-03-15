from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from manifest_lib import ensure_manifest_defaults, load_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import (
    ACCOUNT_IMPORT_WORKFLOW,
    AGENT_DETAIL_ROLE,
    AMPLIFIER_DETAIL_ROLE,
    DISK_DETAIL_ROLE,
    EQUIPMENT_ROLE,
    ROSTER_ROLE,
    UID_PANEL_ROLE,
    filter_records,
    record_head,
    record_screen_role,
    source_index_from_manifest,
)
from roster_taxonomy import canonicalize_agent_label, current_agent_ids


def _reviewed_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return {}
    if str(record.get("qaStatus") or "").lower() != "reviewed":
        return {}
    if not isinstance(labels.get("reviewFinal"), dict):
        return {}
    return labels


def _is_materialized_uid_digit(record: Dict[str, Any], labels: Dict[str, Any]) -> bool:
    if str(record.get("kind") or "").strip() == "derived_uid_digit":
        return True
    head = str(labels.get("head") or record.get("head") or "").strip()
    return head == "uid_digit" and isinstance(labels.get("derivedFromRecordId"), str) and bool(
        str(labels.get("derivedFromRecordId")).strip()
    )


def _manifest_split_index(manifest: Dict[str, Any]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        return output
    for split_name in ("train", "val", "test"):
        record_ids = splits.get(split_name)
        if not isinstance(record_ids, list):
            continue
        for record_id in record_ids:
            output[str(record_id)] = split_name
    return output


def _increment_split(counter: Counter[str], split_name: str) -> None:
    normalized = split_name if split_name in {"train", "val", "test"} else "unsplit"
    counter[normalized] += 1


def _normalize_uid_digit_label(labels: Dict[str, Any]) -> str:
    for key in ("uid_digit", "label"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            digits = "".join(ch for ch in value if ch.isdigit())
            if len(digits) == 1:
                return digits
    return ""


def _normalize_uid_full(labels: Dict[str, Any]) -> str:
    for key in ("uid_full", "label"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            digits = "".join(ch for ch in value if ch.isdigit())
            if 6 <= len(digits) <= 12:
                return digits
    uid_payload = labels.get("uid")
    if isinstance(uid_payload, dict):
        value = uid_payload.get("value")
        if isinstance(value, str) and value.strip():
            digits = "".join(ch for ch in value if ch.isdigit())
            if 6 <= len(digits) <= 12:
                return digits
    return ""


def _normalize_agent_icon_label(labels: Dict[str, Any]) -> str:
    for key in ("agent_icon_id", "agentId", "label"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            canonical = canonicalize_agent_label(value.strip())
            if canonical:
                return canonical
    return ""


def _has_level_pair(labels: Dict[str, Any], current_key: str, cap_key: str) -> bool:
    return isinstance(labels.get(current_key), (int, float)) and isinstance(labels.get(cap_key), (int, float))


def _has_nonempty_mapping(labels: Dict[str, Any], key: str) -> bool:
    value = labels.get(key)
    return isinstance(value, dict) and bool(value)


def _has_nonempty_list(labels: Dict[str, Any], key: str) -> bool:
    value = labels.get(key)
    return isinstance(value, list) and bool(value)


def _split_dict(counter: Counter[str]) -> Dict[str, int]:
    return {name: int(counter.get(name, 0)) for name in ("train", "val", "test", "unsplit")}


def _head_readiness(
    *,
    name: str,
    sample_labels: List[str],
    split_counts: Counter[str],
    min_samples: int,
    min_distinct_labels: int,
    extra_blockers: Iterable[str] = (),
) -> Dict[str, Any]:
    blockers: List[str] = []
    total_samples = len(sample_labels)
    distinct_labels = len(set(sample_labels))
    if total_samples < max(1, int(min_samples)):
        blockers.append(f"need_at_least_{int(min_samples)}_reviewed_samples")
    if distinct_labels < max(1, int(min_distinct_labels)):
        blockers.append(f"need_at_least_{int(min_distinct_labels)}_distinct_labels")
    for split_name in ("train", "val", "test"):
        if int(split_counts.get(split_name, 0)) <= 0:
            blockers.append(f"missing_{split_name}_split_samples")
    blockers.extend(str(item) for item in extra_blockers if str(item).strip())
    return {
        "name": name,
        "sampleCount": total_samples,
        "distinctLabelCount": distinct_labels,
        "splitCounts": _split_dict(split_counts),
        "readyForCurrentTrainPath": not blockers,
        "blockers": blockers,
    }


def _field_target_summary(name: str, split_counts: Counter[str], complete_count: int, total_reviewed_records: int) -> Dict[str, Any]:
    return {
        "name": name,
        "reviewedRecordCount": int(total_reviewed_records),
        "completeRecordCount": int(complete_count),
        "splitCounts": _split_dict(split_counts),
    }


def _markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# OCR Training Readiness")
    lines.append("")
    lines.append(f"- Generated at: `{report['generatedAt']}`")
    lines.append(f"- Workflow: `{report['workflow']}`")
    lines.append(f"- Scoped records: `{report['summary']['scopedRecordCount']}`")
    lines.append(f"- Reviewed truth records: `{report['summary']['reviewedRecordCount']}`")
    lines.append("")
    lines.append("## Current Train Heads")
    lines.append("")
    for key in ("uid_digit", "agent_icon"):
        item = report["trainHeads"][key]
        status = "READY" if item["readyForCurrentTrainPath"] else "BLOCKED"
        lines.append(f"### {item['name']} ({status})")
        lines.append("")
        lines.append(f"- Samples: `{item['sampleCount']}`")
        lines.append(f"- Distinct labels: `{item['distinctLabelCount']}`")
        lines.append(
            f"- Splits: `train={item['splitCounts']['train']}` `val={item['splitCounts']['val']}` `test={item['splitCounts']['test']}` `unsplit={item['splitCounts']['unsplit']}`"
        )
        if key == "uid_digit":
            lines.append(f"- Missing digits: `{', '.join(item['missingDigits']) if item['missingDigits'] else 'none'}`")
            lines.append(f"- Materialized digit samples: `{item['materializedUidDigitSampleCount']}`")
        if key == "agent_icon":
            lines.append(
                f"- Reviewed current agents: `{item['reviewedCurrentAgentCount']}/{item['rosterAgentCount']}`"
            )
            if item["missingReviewedAgents"]:
                lines.append(f"- Missing reviewed agents: `{', '.join(item['missingReviewedAgents'])}`")
        if item["blockers"]:
            lines.append(f"- Blockers: `{'; '.join(item['blockers'])}`")
        lines.append("")

    lines.append("## OCR Field Targets")
    lines.append("")
    for key, item in report["fieldTargets"].items():
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append(f"- Reviewed records: `{item['reviewedRecordCount']}`")
        lines.append(f"- Complete records: `{item['completeRecordCount']}`")
        lines.append(
            f"- Splits: `train={item['splitCounts']['train']}` `val={item['splitCounts']['val']}` `test={item['splitCounts']['test']}` `unsplit={item['splitCounts']['unsplit']}`"
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OCR training readiness report from reviewed truth.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-json", default="docs/training_readiness.json")
    parser.add_argument("--output-md", default="docs/training_readiness.md")
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--import-eligible-only", action="store_true", default=True)
    parser.add_argument("--uid-min-samples", type=int, default=2000)
    parser.add_argument("--agent-icon-min-samples", type=int, default=2000)
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
    split_index = _manifest_split_index(manifest)

    reviewed_records: List[Dict[str, Any]] = []
    reviewed_head_counts: Counter[str] = Counter()
    reviewed_role_counts: Counter[str] = Counter()
    reviewed_split_counts: Counter[str] = Counter()

    uid_digit_labels: List[str] = []
    uid_digit_split_counts: Counter[str] = Counter()
    materialized_uid_digit_samples = 0

    uid_panel_full_uid_records = 0
    uid_panel_full_uid_split_counts: Counter[str] = Counter()

    agent_icon_labels: List[str] = []
    agent_icon_split_counts: Counter[str] = Counter()

    agent_detail_level_total = 0
    agent_detail_level_splits: Counter[str] = Counter()
    agent_detail_mindscape_total = 0
    agent_detail_mindscape_splits: Counter[str] = Counter()
    agent_detail_stats_total = 0
    agent_detail_stats_splits: Counter[str] = Counter()

    amplifier_detail_total = 0
    amplifier_detail_splits: Counter[str] = Counter()
    disk_detail_total = 0
    disk_detail_splits: Counter[str] = Counter()
    equipment_overview_total = 0
    equipment_overview_splits: Counter[str] = Counter()
    roster_owned_total = 0
    roster_owned_splits: Counter[str] = Counter()

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        labels = _reviewed_labels(record)
        if not labels:
            continue
        is_materialized_uid_digit = _is_materialized_uid_digit(record, labels)

        reviewed_records.append(record)
        split_name = split_index.get(str(record.get("id") or ""), "")
        _increment_split(reviewed_split_counts, split_name)

        head = record_head(record)
        role = record_screen_role(record, source_index)
        reviewed_head_counts[head] += 1
        if not is_materialized_uid_digit:
            reviewed_role_counts[role] += 1

        uid_digit = ""
        if is_materialized_uid_digit:
            uid_digit = _normalize_uid_digit_label(labels)
            if uid_digit:
                uid_digit_labels.append(uid_digit)
                _increment_split(uid_digit_split_counts, split_name)
                materialized_uid_digit_samples += 1

        uid_full = _normalize_uid_full(labels)
        if role == UID_PANEL_ROLE and uid_full and not is_materialized_uid_digit:
            uid_panel_full_uid_records += 1
            _increment_split(uid_panel_full_uid_split_counts, split_name)

        agent_icon = _normalize_agent_icon_label(labels)
        if head == "agent_icon" and agent_icon:
            agent_icon_labels.append(agent_icon)
            _increment_split(agent_icon_split_counts, split_name)

        if role == AGENT_DETAIL_ROLE:
            if _has_level_pair(labels, "agent_level", "agent_level_cap"):
                agent_detail_level_total += 1
                _increment_split(agent_detail_level_splits, split_name)
            if _has_level_pair(labels, "agent_mindscape", "agent_mindscape_cap"):
                agent_detail_mindscape_total += 1
                _increment_split(agent_detail_mindscape_splits, split_name)
            if _has_nonempty_mapping(labels, "agent_stats"):
                agent_detail_stats_total += 1
                _increment_split(agent_detail_stats_splits, split_name)

        if role == AMPLIFIER_DETAIL_ROLE:
            if (
                _normalize_agent_icon_label({"agent_icon_id": labels.get("equipment_agent_id")}) or labels.get("equipment_agent_id")
            ) and isinstance(labels.get("amplifier_id"), str) and labels.get("amplifier_id"):
                amplifier_detail_total += 1
                _increment_split(amplifier_detail_splits, split_name)

        if role == DISK_DETAIL_ROLE:
            if isinstance(labels.get("disc_set_id"), str) and labels.get("disc_set_id") and isinstance(labels.get("disc_slot"), (int, float)):
                disk_detail_total += 1
                _increment_split(disk_detail_splits, split_name)

        if role == EQUIPMENT_ROLE:
            if isinstance(labels.get("equipment_agent_id"), str) and labels.get("equipment_agent_id") and _has_nonempty_mapping(labels, "disc_slot_occupancy"):
                equipment_overview_total += 1
                _increment_split(equipment_overview_splits, split_name)

        if role == ROSTER_ROLE and _has_nonempty_list(labels, "owned_agent_ids"):
            roster_owned_total += 1
            _increment_split(roster_owned_splits, split_name)

    current_roster_agents = current_agent_ids()
    reviewed_icon_agents = {label for label in agent_icon_labels if label in set(current_roster_agents)}
    missing_reviewed_agents = [agent for agent in current_roster_agents if agent not in reviewed_icon_agents]
    missing_digits = [str(digit) for digit in range(10) if str(digit) not in set(uid_digit_labels)]

    uid_digit_head = _head_readiness(
        name="uid_digit",
        sample_labels=uid_digit_labels,
        split_counts=uid_digit_split_counts,
        min_samples=args.uid_min_samples,
        min_distinct_labels=10,
        extra_blockers=[],
    )
    uid_digit_head["missingDigits"] = missing_digits
    uid_digit_head["materializedUidDigitSampleCount"] = int(materialized_uid_digit_samples)
    uid_digit_head["reviewedUidPanelTruthCount"] = int(uid_panel_full_uid_records)
    if uid_panel_full_uid_records > 0:
        uid_digit_head["notes"] = [
            "Reviewed uid_panel source records hold full UID truth; trainable per-digit samples come from materialized uid_digit crops.",
        ]

    agent_icon_head = _head_readiness(
        name="agent_icon",
        sample_labels=agent_icon_labels,
        split_counts=agent_icon_split_counts,
        min_samples=args.agent_icon_min_samples,
        min_distinct_labels=2,
        extra_blockers=[],
    )
    agent_icon_head["rosterAgentCount"] = len(current_roster_agents)
    agent_icon_head["reviewedCurrentAgentCount"] = len(reviewed_icon_agents)
    agent_icon_head["missingReviewedAgents"] = missing_reviewed_agents
    if missing_reviewed_agents:
        agent_icon_head["blockers"].append("missing_current_roster_agent_coverage")
        agent_icon_head["readyForCurrentTrainPath"] = False

    report = {
        "generatedAt": utc_now(),
        "workflow": str(args.workflow),
        "importEligibleOnly": bool(args.import_eligible_only),
        "summary": {
            "scopedRecordCount": len(scoped_records),
            "reviewedRecordCount": len(reviewed_records),
            "reviewedHeadCounts": dict(reviewed_head_counts),
            "reviewedScreenRoleCounts": dict(reviewed_role_counts),
            "reviewedSplitCounts": _split_dict(reviewed_split_counts),
        },
        "trainHeads": {
            "uid_digit": uid_digit_head,
            "agent_icon": agent_icon_head,
        },
        "fieldTargets": {
            "uid_panel_full_uid": _field_target_summary(
                "uid_panel_full_uid",
                uid_panel_full_uid_split_counts,
                uid_panel_full_uid_records,
                uid_panel_full_uid_records,
            ),
            "agent_detail_level": _field_target_summary(
                "agent_detail_level",
                agent_detail_level_splits,
                agent_detail_level_total,
                reviewed_role_counts.get(AGENT_DETAIL_ROLE, 0),
            ),
            "agent_detail_mindscape": _field_target_summary(
                "agent_detail_mindscape",
                agent_detail_mindscape_splits,
                agent_detail_mindscape_total,
                reviewed_role_counts.get(AGENT_DETAIL_ROLE, 0),
            ),
            "agent_detail_stats": _field_target_summary(
                "agent_detail_stats",
                agent_detail_stats_splits,
                agent_detail_stats_total,
                reviewed_role_counts.get(AGENT_DETAIL_ROLE, 0),
            ),
            "amplifier_detail": _field_target_summary(
                "amplifier_detail",
                amplifier_detail_splits,
                amplifier_detail_total,
                reviewed_role_counts.get(AMPLIFIER_DETAIL_ROLE, 0),
            ),
            "disk_detail": _field_target_summary(
                "disk_detail",
                disk_detail_splits,
                disk_detail_total,
                reviewed_role_counts.get(DISK_DETAIL_ROLE, 0),
            ),
            "equipment_overview": _field_target_summary(
                "equipment_overview",
                equipment_overview_splits,
                equipment_overview_total,
                reviewed_role_counts.get(EQUIPMENT_ROLE, 0),
            ),
            "roster_owned_agents": _field_target_summary(
                "roster_owned_agents",
                roster_owned_splits,
                roster_owned_total,
                reviewed_role_counts.get(ROSTER_ROLE, 0),
            ),
        },
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    output_md = Path(args.output_md).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown_report(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
