from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equipment_taxonomy import normalize_name_and_id, valid_stat_key
from ocr_dataset_policy import (
    AGENT_DETAIL_ROLE,
    AMPLIFIER_DETAIL_ROLE,
    DISK_DETAIL_ROLE,
    EQUIPMENT_ROLE,
    MINDSCAPE_ROLE,
    UID_PANEL_ROLE,
)
from roster_taxonomy import canonicalize_agent_label


def _text(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _collect_row_payload(row: Dict[str, Any]) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for key, value in row.items():
        text = _text(value)
        if text:
            payload[str(key)] = text
    return payload


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _coerce_number(raw: Any) -> int | float | None:
    text = _text(raw)
    if not text:
        return None
    normalized = text.replace("%", "").replace("+", "").replace(",", ".").replace(" ", "")
    value = float(normalized)
    if value.is_integer():
        return int(value)
    return round(value, 4)


def _parse_int(raw: Any, *, field_name: str, min_value: int | None = None, max_value: int | None = None, fallback: Any = "") -> int | None:
    value = _coerce_number(raw if _text(raw) else fallback)
    if value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field_name} must be an integer: {raw!r}")
    result = int(value)
    if min_value is not None and result < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}: {result}")
    if max_value is not None and result > max_value:
        raise ValueError(f"{field_name} must be <= {max_value}: {result}")
    return result


def _parse_json(raw: Any, *, field_name: str, expected_type: type, fallback: Any = None) -> Any:
    text = _text(raw)
    if not text:
        return fallback
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if not isinstance(payload, expected_type):
        raise ValueError(f"{field_name} must decode to {expected_type.__name__}")
    return payload


def _normalize_uid_full(raw: Any, fallback: Any = "") -> str:
    value = _first_text(raw, fallback)
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        raise ValueError("uid_full is required")
    if not 6 <= len(digits) <= 12:
        raise ValueError(f"uid_full must have 6-12 digits: {digits}")
    return digits


def _normalize_agent_icon(raw: Any, fallback: Any = "") -> str:
    value = _first_text(raw, fallback)
    canonical = canonicalize_agent_label(value)
    if not canonical or canonical == "unknown":
        raise ValueError(f"Invalid agent_icon_id value: {value!r}")
    return canonical


def _normalize_level_pair(current_raw: Any, max_raw: Any, *, field_prefix: str, fallback_current: Any = "", fallback_max: Any = "") -> Tuple[int | None, int | None]:
    current = _parse_int(current_raw, field_name=f"{field_prefix}_current", min_value=0, max_value=99, fallback=fallback_current)
    level_cap = _parse_int(max_raw, field_name=f"{field_prefix}_max", min_value=0, max_value=99, fallback=fallback_max)
    if current is None and level_cap is None:
        return None, None
    if current is None or level_cap is None:
        raise ValueError(f"{field_prefix} pair must include both current and max")
    if current > level_cap:
        raise ValueError(f"{field_prefix}_current must be <= {field_prefix}_max")
    return current, level_cap


def _normalize_stat_key(raw: Any, *, field_name: str) -> str:
    value = _text(raw)
    if not value or not valid_stat_key(value):
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return value


def _normalize_stat_map(raw: Any, fallback: Any = None) -> Dict[str, int | float]:
    payload = _parse_json(raw, field_name="agent_stats_json", expected_type=dict, fallback=fallback)
    if not payload:
        return {}
    output: Dict[str, int | float] = {}
    for key, value in payload.items():
        normalized_key = _normalize_stat_key(key, field_name="agent_stats_json.key")
        normalized_value = _coerce_number(value)
        if normalized_value is None:
            raise ValueError(f"Invalid value for agent_stats_json[{normalized_key!r}]")
        output[normalized_key] = normalized_value
    return output


def _normalize_substats(raw: Any, fallback: Any = None) -> List[Dict[str, int | float | str]]:
    payload = _parse_json(raw, field_name="disc_substats_json", expected_type=list, fallback=fallback)
    if not payload:
        return []
    output: List[Dict[str, int | float | str]] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"disc_substats_json[{index}] must be an object")
        key = _normalize_stat_key(entry.get("key"), field_name=f"disc_substats_json[{index}].key")
        value = _coerce_number(entry.get("value"))
        if value is None:
            raise ValueError(f"disc_substats_json[{index}].value must be numeric")
        output.append({"key": key, "value": value})
    return output


def _normalize_equipment_summary(raw: Any, fallback: Any = None) -> List[Dict[str, Any]]:
    payload = _parse_json(raw, field_name="equipment_disc_summary_json", expected_type=list, fallback=fallback)
    if not payload:
        return []
    output: List[Dict[str, Any]] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"equipment_disc_summary_json[{index}] must be an object")
        slot = _parse_int(entry.get("slot"), field_name=f"equipment_disc_summary_json[{index}].slot", min_value=1, max_value=6)
        if slot is None:
            raise ValueError(f"equipment_disc_summary_json[{index}].slot is required")
        set_id, set_name = normalize_name_and_id(
            entry.get("setName"),
            entry.get("setId"),
            kind="disc_set",
        )
        level, level_cap = _normalize_level_pair(
            entry.get("level"),
            entry.get("levelCap"),
            field_prefix=f"equipment_disc_summary_json[{index}].level",
        )
        item: Dict[str, Any] = {
            "slot": slot,
            "setId": set_id,
            "displayName": set_name,
        }
        if level is not None:
            item["level"] = level
        if level_cap is not None:
            item["levelCap"] = level_cap
        output.append(item)
    output.sort(key=lambda item: int(item["slot"]))
    return output


def _review_metadata(row: Dict[str, Any], reviewer_id: str) -> Dict[str, str]:
    return {
        "reviewer": _first_text(row.get("reviewer"), reviewer_id) or reviewer_id,
        "reviewedAt": utc_now(),
        "notes": _text(row.get("notes")),
    }


def _detect_head(record: Dict[str, Any], labels: Dict[str, Any], suggested: Dict[str, Any]) -> str:
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    for payload in (labels, suggested):
        maybe = payload.get("head")
        if isinstance(maybe, str) and maybe.strip():
            return maybe.strip()
    return "unknown"


def _screen_role(record: Dict[str, Any]) -> str:
    value = record.get("screenRole")
    return value.strip() if isinstance(value, str) else ""


def _apply_uid_panel(labels: Dict[str, Any], row: Dict[str, Any], suggested: Dict[str, Any]) -> None:
    uid_full = _normalize_uid_full(row.get("uid_full"), labels.get("uid_full") or suggested.get("uid_full") or suggested.get("uid_digit"))
    labels["uid_full"] = uid_full
    labels["uid"] = {"value": uid_full}
    labels["label"] = uid_full


def _apply_agent_detail(labels: Dict[str, Any], row: Dict[str, Any], suggested: Dict[str, Any]) -> None:
    agent_id = _normalize_agent_icon(row.get("agent_icon_id"), labels.get("agent_icon_id") or suggested.get("agent_icon_id"))
    level, level_cap = _normalize_level_pair(
        row.get("agent_level_current"),
        row.get("agent_level_max"),
        field_prefix="agent_level",
        fallback_current=labels.get("agent_level"),
        fallback_max=labels.get("agent_level_cap"),
    )
    mindscape, mindscape_cap = _normalize_level_pair(
        row.get("agent_mindscape_current"),
        row.get("agent_mindscape_max"),
        field_prefix="agent_mindscape",
        fallback_current=labels.get("agent_mindscape"),
        fallback_max=labels.get("agent_mindscape_cap"),
    )
    stats = _normalize_stat_map(row.get("agent_stats_json"), fallback=labels.get("agent_stats"))

    labels["agent_icon_id"] = agent_id
    labels["label"] = agent_id
    if level is not None:
        labels["agent_level"] = level
    if level_cap is not None:
        labels["agent_level_cap"] = level_cap
    if mindscape is not None:
        labels["agent_mindscape"] = mindscape
    if mindscape_cap is not None:
        labels["agent_mindscape_cap"] = mindscape_cap
    if stats:
        labels["agent_stats"] = stats

    snapshot: Dict[str, Any] = {"agentId": agent_id}
    if level is not None:
        snapshot["level"] = level
    if level_cap is not None:
        snapshot["levelCap"] = level_cap
    if mindscape is not None:
        snapshot["mindscape"] = mindscape
    if mindscape_cap is not None:
        snapshot["mindscapeCap"] = mindscape_cap
    if stats:
        snapshot["stats"] = stats
    labels["agentSnapshot"] = snapshot


def _apply_amplifier_detail(labels: Dict[str, Any], row: Dict[str, Any]) -> None:
    amplifier_id, amplifier_name = normalize_name_and_id(
        row.get("amplifier_name"),
        row.get("amplifier_id"),
        kind="amplifier",
    )
    level, level_cap = _normalize_level_pair(
        row.get("amplifier_level_current"),
        row.get("amplifier_level_max"),
        field_prefix="amplifier_level",
        fallback_current=labels.get("amplifier_level"),
        fallback_max=labels.get("amplifier_level_cap"),
    )
    base_stat_key = _normalize_stat_key(row.get("amplifier_base_stat_key"), field_name="amplifier_base_stat_key")
    base_stat_value = _coerce_number(row.get("amplifier_base_stat_value"))
    advanced_stat_key = _normalize_stat_key(row.get("amplifier_advanced_stat_key"), field_name="amplifier_advanced_stat_key")
    advanced_stat_value = _coerce_number(row.get("amplifier_advanced_stat_value"))
    if base_stat_value is None or advanced_stat_value is None:
        raise ValueError("Amplifier stat values must be numeric")

    owner_agent_id = _text(row.get("equipment_agent_id"))
    if owner_agent_id:
        owner_agent_id = _normalize_agent_icon(owner_agent_id)

    labels["amplifier_id"] = amplifier_id
    labels["amplifier_name"] = amplifier_name
    labels["label"] = amplifier_id
    if level is not None:
        labels["amplifier_level"] = level
    if level_cap is not None:
        labels["amplifier_level_cap"] = level_cap
    labels["amplifier_base_stat_key"] = base_stat_key
    labels["amplifier_base_stat_value"] = base_stat_value
    labels["amplifier_advanced_stat_key"] = advanced_stat_key
    labels["amplifier_advanced_stat_value"] = advanced_stat_value
    if owner_agent_id:
        labels["equipment_agent_id"] = owner_agent_id

    weapon: Dict[str, Any] = {
        "weaponId": amplifier_id,
        "displayName": amplifier_name,
        "baseStatKey": base_stat_key,
        "baseStatValue": base_stat_value,
        "advancedStatKey": advanced_stat_key,
        "advancedStatValue": advanced_stat_value,
    }
    if level is not None:
        weapon["level"] = level
    if level_cap is not None:
        weapon["levelCap"] = level_cap
    if owner_agent_id:
        weapon["agentId"] = owner_agent_id
    labels["weapon"] = weapon


def _apply_disk_detail(labels: Dict[str, Any], row: Dict[str, Any]) -> None:
    slot = _parse_int(row.get("disc_slot"), field_name="disc_slot", min_value=1, max_value=6)
    if slot is None:
        raise ValueError("disc_slot is required")
    disc_set_id, disc_set_name = normalize_name_and_id(
        row.get("disc_set_name"),
        row.get("disc_set_id"),
        kind="disc_set",
    )
    level, level_cap = _normalize_level_pair(
        row.get("disc_level_current"),
        row.get("disc_level_max"),
        field_prefix="disc_level",
        fallback_current=labels.get("disc_level"),
        fallback_max=labels.get("disc_level_cap"),
    )
    main_stat_key = _normalize_stat_key(row.get("disc_main_stat_key"), field_name="disc_main_stat_key")
    main_stat_value = _coerce_number(row.get("disc_main_stat_value"))
    if main_stat_value is None:
        raise ValueError("disc_main_stat_value must be numeric")
    substats = _normalize_substats(row.get("disc_substats_json"), fallback=labels.get("disc_substats"))

    owner_agent_id = _text(row.get("equipment_agent_id"))
    if owner_agent_id:
        owner_agent_id = _normalize_agent_icon(owner_agent_id)

    labels["disc_slot"] = slot
    labels["disc_set_id"] = disc_set_id
    labels["disc_set_name"] = disc_set_name
    labels["disc_level"] = level
    labels["disc_level_cap"] = level_cap
    labels["disc_main_stat_key"] = main_stat_key
    labels["disc_main_stat_value"] = main_stat_value
    labels["disc_substats"] = substats
    labels["label"] = disc_set_id
    if owner_agent_id:
        labels["equipment_agent_id"] = owner_agent_id

    disc: Dict[str, Any] = {
        "slot": slot,
        "setId": disc_set_id,
        "displayName": disc_set_name,
        "mainStatKey": main_stat_key,
        "mainStatValue": main_stat_value,
        "substats": substats,
    }
    if level is not None:
        disc["level"] = level
    if level_cap is not None:
        disc["levelCap"] = level_cap
    if owner_agent_id:
        disc["agentId"] = owner_agent_id
    labels["disc"] = disc


def _apply_equipment_overview(labels: Dict[str, Any], row: Dict[str, Any]) -> None:
    owner_agent_id = _normalize_agent_icon(row.get("equipment_agent_id"), labels.get("equipment_agent_id") or labels.get("agent_icon_id"))
    summary = _normalize_equipment_summary(row.get("equipment_disc_summary_json"), fallback=labels.get("equipment_disc_summary"))
    amplifier_name = _text(row.get("amplifier_name"))
    amplifier_id = _text(row.get("amplifier_id"))
    weapon: Dict[str, Any] = {}
    if amplifier_name or amplifier_id:
        normalized_id, normalized_name = normalize_name_and_id(amplifier_name, amplifier_id, kind="amplifier")
        labels["amplifier_id"] = normalized_id
        labels["amplifier_name"] = normalized_name
        weapon = {"weaponId": normalized_id, "displayName": normalized_name}

    labels["equipment_agent_id"] = owner_agent_id
    labels["equipment_disc_summary"] = summary
    labels["label"] = owner_agent_id

    overview: Dict[str, Any] = {"agentId": owner_agent_id, "discs": summary}
    if weapon:
        overview["weapon"] = weapon
    labels["equipmentOverview"] = overview


def _apply_provenance_only(labels: Dict[str, Any], role: str) -> None:
    labels["label"] = "provenance_only"
    labels["provenanceOnly"] = True
    if role == MINDSCAPE_ROLE:
        labels["derivedFrom"] = "agent_detail"
        labels["reason"] = "mindscape_derived_from_agent_detail_footer"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed OCR labels from CSV to dataset manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--review-round", choices=["A", "B", "final"], default="final")
    parser.add_argument("--reviewer-id", default="human")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    record_map = {str(record.get("id") or ""): record for record in records if isinstance(record, dict)}

    applied = 0
    missing = 0
    invalid = 0
    with Path(args.input_csv).resolve().open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record_id = _text(row.get("record_id"))
            if not record_id:
                continue
            record = record_map.get(record_id)
            if record is None:
                missing += 1
                continue

            labels = record.get("labels")
            if not isinstance(labels, dict):
                labels = {}
                record["labels"] = labels
            suggested = record.get("suggestedLabels")
            if not isinstance(suggested, dict):
                suggested = {}

            if args.review_round in {"A", "B"}:
                labels[f"reviewer{args.review_round}"] = {
                    **_collect_row_payload(row),
                    "reviewer": _first_text(row.get("reviewer"), args.reviewer_id) or args.reviewer_id,
                    "reviewedAt": utc_now(),
                }
                record["qaStatus"] = "needs_review"
                applied += 1
                continue

            head = _detect_head(record, labels, suggested)
            role = _screen_role(record)
            review_meta = _review_metadata(row, args.reviewer_id)
            try:
                if _text(row.get("record_disposition")) == "provenance_only":
                    _apply_provenance_only(labels, role)
                elif role == UID_PANEL_ROLE or head == "uid_digit":
                    _apply_uid_panel(labels, row, suggested)
                elif role == AGENT_DETAIL_ROLE:
                    _apply_agent_detail(labels, row, suggested)
                elif role == AMPLIFIER_DETAIL_ROLE:
                    _apply_amplifier_detail(labels, row)
                elif role == DISK_DETAIL_ROLE:
                    _apply_disk_detail(labels, row)
                elif role == EQUIPMENT_ROLE:
                    _apply_equipment_overview(labels, row)
                else:
                    raise ValueError(f"Unsupported review target for screenRole={role!r}, head={head!r}")
            except ValueError:
                invalid += 1
                record["qaStatus"] = "needs_review"
                continue

            labels["head"] = head
            labels["reviewFinal"] = review_meta
            record["qaStatus"] = "reviewed"
            applied += 1

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["humanReview"] = "completed" if args.review_round == "final" else "in_progress"
        qa_status["qaPass2"] = "completed" if args.review_round == "final" else "pending"
        qa_status["qaUpdatedAt"] = utc_now()

    save_manifest(manifest_path, manifest)
    print(f"Applied labels: {applied}; missing records: {missing}; invalid rows: {invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
