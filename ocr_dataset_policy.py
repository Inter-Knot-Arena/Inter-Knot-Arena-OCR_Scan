from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Set

ACCOUNT_IMPORT_WORKFLOW = "account_import"
COMBAT_REFERENCE_WORKFLOW = "combat_reference"
ANY_WORKFLOW = "any"

UID_PANEL_ROLE = "uid_panel"
ROSTER_ROLE = "roster"
AGENT_DETAIL_ROLE = "agent_detail"
EQUIPMENT_ROLE = "equipment"
DISK_DETAIL_ROLE = "disk_detail"
AMPLIFIER_DETAIL_ROLE = "amplifier_detail"
MINDSCAPE_ROLE = "mindscape"
ACCOUNT_UNKNOWN_ROLE = "account_unknown"
COMBAT_ROLE = "combat"
OTHER_ROLE = "other"

HEAD_TO_DEFAULT_ROLE = {
    "uid_digit": UID_PANEL_ROLE,
    "agent_icon": AGENT_DETAIL_ROLE,
    "equipment": EQUIPMENT_ROLE,
}

ROLE_TO_ELIGIBLE_HEADS = {
    UID_PANEL_ROLE: {"uid_digit"},
    ROSTER_ROLE: {"agent_icon"},
    AGENT_DETAIL_ROLE: {"agent_icon"},
    EQUIPMENT_ROLE: {"equipment", "agent_icon"},
    DISK_DETAIL_ROLE: {"equipment"},
    AMPLIFIER_DETAIL_ROLE: {"equipment"},
    MINDSCAPE_ROLE: set(),
    ACCOUNT_UNKNOWN_ROLE: {"uid_digit", "agent_icon", "equipment"},
    COMBAT_ROLE: set(),
    OTHER_ROLE: set(),
}


def _text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    for item in values:
        if isinstance(item, str):
            value = item.strip()
            if value:
                output.append(value)
    return output


def default_role_for_head(head: str) -> str:
    return HEAD_TO_DEFAULT_ROLE.get(str(head or "").strip(), ACCOUNT_UNKNOWN_ROLE)


def infer_source_workflow(source: Dict[str, Any]) -> str:
    workflow = str(source.get("workflow") or "").strip()
    if workflow:
        return workflow

    source_id = str(source.get("sourceId") or "").strip()
    source_type = str(source.get("sourceType") or "").strip().lower()
    tags = {item.lower() for item in _text_list(source.get("sourceTags"))}

    if source_id.startswith("live_session_"):
        return ACCOUNT_IMPORT_WORKFLOW
    if {
        "profile",
        "roster",
        "account",
        "inventory",
        "equipment",
        "uid",
        "agent_detail",
        "stats",
        "disc",
        "disk",
        "disc_detail",
        "disk_detail",
        "amplifier",
        "amplificator",
        "w_engine",
        "mindscape",
        "cinema",
    } & tags:
        return ACCOUNT_IMPORT_WORKFLOW
    if source_type == "live-capture":
        return ACCOUNT_IMPORT_WORKFLOW
    return COMBAT_REFERENCE_WORKFLOW


def infer_source_screen_role(source: Dict[str, Any]) -> str:
    screen_role = str(source.get("screenRole") or "").strip()
    if screen_role:
        return screen_role

    tags = {item.lower() for item in _text_list(source.get("sourceTags"))}
    if "uid" in tags or "profile" in tags:
        return UID_PANEL_ROLE
    if "roster" in tags:
        return ROSTER_ROLE
    if "mindscape" in tags or "cinema" in tags:
        return MINDSCAPE_ROLE
    if "amplifier" in tags or "amplificator" in tags or "w_engine" in tags:
        return AMPLIFIER_DETAIL_ROLE
    if "disc_detail" in tags or "disk_detail" in tags or "disc" in tags or "disk" in tags:
        return DISK_DETAIL_ROLE
    if "equipment" in tags or "inventory" in tags:
        return EQUIPMENT_ROLE
    if "agent_detail" in tags or "agent" in tags or "stats" in tags:
        return AGENT_DETAIL_ROLE
    if infer_source_workflow(source) == ACCOUNT_IMPORT_WORKFLOW:
        return ACCOUNT_UNKNOWN_ROLE
    return COMBAT_ROLE


def infer_source_eligible_heads(source: Dict[str, Any]) -> list[str]:
    explicit = _text_list(source.get("eligibleHeads"))
    if explicit:
        return explicit
    role = infer_source_screen_role(source)
    heads = sorted(ROLE_TO_ELIGIBLE_HEADS.get(role, set()))
    return heads


def apply_source_policy_defaults(source: Dict[str, Any]) -> Dict[str, Any]:
    source["workflow"] = infer_source_workflow(source)
    source["screenRole"] = infer_source_screen_role(source)
    source["eligibleHeads"] = infer_source_eligible_heads(source)
    return source


def record_head(record: Dict[str, Any]) -> str:
    head = record.get("head")
    if isinstance(head, str) and head.strip():
        return head.strip()
    for key in ("labels", "suggestedLabels"):
        payload = record.get(key)
        if not isinstance(payload, dict):
            continue
        value = payload.get("head")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def record_workflow(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> str:
    workflow = record.get("workflow")
    if isinstance(workflow, str) and workflow.strip():
        return workflow.strip()
    source = source_index.get(str(record.get("sourceId") or "").strip(), {})
    if isinstance(source, dict):
        return infer_source_workflow(source)
    return COMBAT_REFERENCE_WORKFLOW


def record_screen_role(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> str:
    screen_role = record.get("screenRole")
    if isinstance(screen_role, str) and screen_role.strip():
        return screen_role.strip()
    source = source_index.get(str(record.get("sourceId") or "").strip(), {})
    if isinstance(source, dict):
        role = infer_source_screen_role(source)
        if role != ACCOUNT_UNKNOWN_ROLE:
            return role
    return default_role_for_head(record_head(record))


def record_eligible_heads(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> Set[str]:
    explicit = _text_list(record.get("eligibleHeads"))
    if explicit:
        return set(explicit)
    source = source_index.get(str(record.get("sourceId") or "").strip(), {})
    if isinstance(source, dict):
        eligible = infer_source_eligible_heads(source)
        if eligible:
            return set(eligible)
    return set(ROLE_TO_ELIGIBLE_HEADS.get(record_screen_role(record, source_index), set()))


def apply_record_policy_defaults(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    record["workflow"] = record_workflow(record, source_index)
    record["screenRole"] = record_screen_role(record, source_index)
    record["eligibleHeads"] = sorted(record_eligible_heads(record, source_index))
    record["ocrImportEligible"] = bool(
        record["workflow"] == ACCOUNT_IMPORT_WORKFLOW and record_head(record) in set(record["eligibleHeads"])
    )
    return record


def record_matches_workflow(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]], workflow: str) -> bool:
    normalized = str(workflow or ANY_WORKFLOW).strip().lower()
    if normalized in {"", ANY_WORKFLOW}:
        return True
    return record_workflow(record, source_index).lower() == normalized


def record_is_import_eligible(record: Dict[str, Any], source_index: Dict[str, Dict[str, Any]]) -> bool:
    if record_workflow(record, source_index) != ACCOUNT_IMPORT_WORKFLOW:
        return False
    head = record_head(record)
    return head in record_eligible_heads(record, source_index)


def filter_records(
    records: Iterable[Dict[str, Any]],
    source_index: Dict[str, Dict[str, Any]],
    workflow: str = ACCOUNT_IMPORT_WORKFLOW,
    import_eligible_only: bool = True,
) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if not record_matches_workflow(record, source_index, workflow):
            continue
        if import_eligible_only and not record_is_import_eligible(record, source_index):
            continue
        output.append(record)
    return output


def source_index_from_manifest(sources: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(sources, list):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("sourceId") or "").strip()
        if not source_id:
            continue
        output[source_id] = source
    return output
