from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_review_agent_detail_level import (  # noqa: E402
    _build_digit_templates,
    _connected_components,
    _extract_reference_value_block,
    _load_image_gray,
    _merge_overlapping_boxes,
    _normalize_glyph,
)
from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, AGENT_DETAIL_ROLE, filter_records, record_screen_role, source_index_from_manifest  # noqa: E402


FIELD_SPECS: list[dict[str, Any]] = [
    {"source": "hp", "canonical": "hp_flat", "kind": "int", "min": 1000, "max": 30000},
    {"source": "atk", "canonical": "attack_flat", "kind": "int", "min": 100, "max": 6000},
    {"source": "def", "canonical": "defense_flat", "kind": "int", "min": 100, "max": 4000},
    {"source": "impact", "canonical": "impact", "kind": "int", "min": 1, "max": 400},
    {"source": "crit_rate", "canonical": "crit_rate_pct", "kind": "percent", "min": 0, "max": 100},
    {"source": "crit_dmg", "canonical": "crit_damage_pct", "kind": "percent", "min": 0, "max": 400},
    {"source": "anomaly_mastery", "canonical": "anomaly_mastery", "kind": "int", "min": 0, "max": 600},
    {"source": "anomaly_prof", "canonical": "anomaly_proficiency", "kind": "int", "min": 0, "max": 600},
    {"source": "pen_ratio", "canonical": "pen_ratio_pct", "kind": "percent", "min": 0, "max": 100},
    {"source": "energy_regen", "canonical": "energy_regen", "kind": "decimal", "min": 0, "max": 100},
]

MAX_DIGIT_SCORE = 0.17
MIN_DIGIT_MARGIN = 0.01


def _reviewed_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return {}
    if str(record.get("qaStatus") or "").strip().lower() != "reviewed":
        return {}
    if not isinstance(labels.get("reviewFinal"), dict):
        return {}
    return labels


def _classify_digit(image, templates: Mapping[str, list]) -> tuple[str, float, float]:
    sample = _normalize_glyph(image)
    scores: list[tuple[float, str]] = []
    for label, variants in templates.items():
        if not variants:
            continue
        best_score = min(float(((sample - variant) ** 2).mean()) for variant in variants)
        scores.append((best_score, label))
    if not scores:
        return "", float("inf"), 0.0
    scores.sort(key=lambda item: item[0])
    best_score, best_label = scores[0]
    margin = float(scores[1][0] - best_score) if len(scores) > 1 else 1.0
    return best_label, float(best_score), margin


def _glyph_infos(block, templates: Mapping[str, list]) -> list[dict[str, Any]]:
    components = _connected_components(block, area_min=10, height_min=6, width_min=1)
    merged_boxes = _merge_overlapping_boxes(components)
    infos: list[dict[str, Any]] = []
    for x0, y0, x1, y1 in merged_boxes:
        image = block[y0:y1, x0:x1]
        label, score, margin = _classify_digit(image, templates)
        height, width = image.shape[:2]
        infos.append(
            {
                "x": int(x0),
                "y": int(y0),
                "height": int(height),
                "width": int(width),
                "digit": label,
                "score": score,
                "margin": margin,
            }
        )
    return infos


def _parse_field(block, spec: Mapping[str, Any], templates: Mapping[str, list]) -> tuple[int | float | None, list[dict[str, Any]], str | None]:
    infos = _glyph_infos(block, templates)
    if not infos:
        return None, infos, "no_glyphs"

    median_height = statistics.median(info["height"] for info in infos)
    median_width = statistics.median(info["width"] for info in infos)
    median_y = statistics.median(info["y"] for info in infos)
    expected_kind = str(spec["kind"])

    tokens: list[dict[str, Any]] = []
    for index, info in enumerate(infos):
        is_last = index == (len(infos) - 1)
        is_separator = (
            info["height"] <= median_height * 0.68
            and info["width"] <= max(9.0, median_width * 0.62)
            and info["y"] >= median_y + median_height * 0.35
        )
        looks_percent = (
            expected_kind == "percent"
            and is_last
            and info["height"] >= median_height * 0.82
            and info["width"] >= median_width * 1.12
        )
        token = dict(info)
        if is_separator:
            token["type"] = "sep"
        elif looks_percent:
            token["type"] = "percent"
        else:
            token["type"] = "digit"
        tokens.append(token)

    if expected_kind == "int":
        # Tiny low-hanging artifacts occasionally leak into integer rows; ignore them instead of rejecting the row.
        tokens = [token for token in tokens if token["type"] != "sep"]

    digits = [str(token["digit"]) for token in tokens if token["type"] == "digit"]
    if any(not digit.isdigit() for digit in digits):
        return None, tokens, "nondigit"

    separator_count = sum(1 for token in tokens if token["type"] == "sep")
    percent_count = sum(1 for token in tokens if token["type"] == "percent")

    if expected_kind == "int":
        if percent_count:
            return None, tokens, "bad_token_int"
        return int("".join(digits)), tokens, None

    if expected_kind == "decimal":
        if percent_count or separator_count != 1 or len(digits) < 2:
            return None, tokens, "bad_token_decimal"
        return float(f"{''.join(digits[:-1])}.{digits[-1]}"), tokens, None

    if expected_kind == "percent":
        if percent_count > 1 or separator_count > 1:
            return None, tokens, "bad_token_percent"
        if separator_count == 1:
            if len(digits) < 2:
                return None, tokens, "few_digits_percent"
            return float(f"{''.join(digits[:-1])}.{digits[-1]}"), tokens, None
        if not digits:
            return None, tokens, "no_digits_percent"
        return int("".join(digits)), tokens, None

    return None, tokens, "unsupported_kind"


def _update_snapshot(labels: Dict[str, Any], stats: Dict[str, int | float]) -> None:
    snapshot = labels.get("agentSnapshot")
    snapshot = dict(snapshot) if isinstance(snapshot, dict) else {}
    agent_id = str(labels.get("agent_icon_id") or "").strip()
    if agent_id:
        snapshot["agentId"] = agent_id
    if labels.get("agent_level") is not None:
        snapshot["level"] = labels.get("agent_level")
    if labels.get("agent_level_cap") is not None:
        snapshot["levelCap"] = labels.get("agent_level_cap")
    if labels.get("agent_mindscape") is not None:
        snapshot["mindscape"] = labels.get("agent_mindscape")
    if labels.get("agent_mindscape_cap") is not None:
        snapshot["mindscapeCap"] = labels.get("agent_mindscape_cap")
    snapshot["stats"] = stats
    labels["agentSnapshot"] = snapshot


def _update_review_final(labels: Dict[str, Any], reviewer_id: str) -> None:
    review_final = labels.get("reviewFinal")
    review_final = dict(review_final) if isinstance(review_final, dict) else {}
    note = "Auto-reviewed agent stats from detected right-panel stat rows."
    prior_notes = str(review_final.get("notes") or "").strip()
    if note not in prior_notes:
        review_final["notes"] = f"{prior_notes} {note}".strip()
    review_final["reviewer"] = reviewer_id
    review_final["reviewedAt"] = utc_now()
    labels["reviewFinal"] = review_final


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-review agent_detail stat maps from right-panel stat rows.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--workflow", default=ACCOUNT_IMPORT_WORKFLOW)
    parser.add_argument("--reviewer-id", default="auto_review_agent_detail_stats")
    parser.add_argument("--output-json", default="docs/auto_review_agent_detail_stats.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    source_index = source_index_from_manifest(manifest.get("sources", []))
    scoped_records = filter_records(
        records,
        source_index=source_index,
        workflow=args.workflow,
        import_eligible_only=True,
    )

    templates = _build_digit_templates()
    if len(templates) < 10:
        raise RuntimeError(f"Insufficient digit template coverage: {sorted(templates.keys())}")

    processed = 0
    applied = 0
    skipped_existing = 0
    skipped_missing_assets = 0
    skipped_nonreviewed = 0
    skipped_derived = 0
    skipped_parse_failed = 0
    failure_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for record in scoped_records:
        if not isinstance(record, dict):
            continue
        if record_screen_role(record, source_index) != AGENT_DETAIL_ROLE:
            continue
        if str(record.get("kind") or "").strip().lower() != "manual_screenshot":
            skipped_derived += 1
            continue

        labels = _reviewed_labels(record)
        if not labels:
            skipped_nonreviewed += 1
            continue
        if isinstance(labels.get("agent_stats"), dict) and labels["agent_stats"]:
            skipped_existing += 1
            continue

        path_value = str(record.get("path") or "").strip()
        if not path_value:
            skipped_missing_assets += 1
            continue
        image = _load_image_gray(Path(path_value))
        if image is None:
            skipped_missing_assets += 1
            continue

        processed += 1
        parsed_stats: dict[str, int | float] = {}
        failure_reason = ""
        digit_scores: list[float] = []
        digit_margins: list[float] = []

        for spec in FIELD_SPECS:
            block = _extract_reference_value_block(image, str(spec["source"]))
            if block is None:
                failure_reason = f"{spec['source']}:no_block"
                break

            value, tokens, error = _parse_field(block, spec, templates)
            if error:
                failure_reason = f"{spec['source']}:{error}"
                break

            digit_tokens = [token for token in tokens if token["type"] == "digit"]
            if not digit_tokens:
                failure_reason = f"{spec['source']}:no_digits"
                break

            max_score = max(float(token["score"]) for token in digit_tokens)
            min_margin = min(float(token["margin"]) for token in digit_tokens)
            if max_score > MAX_DIGIT_SCORE or min_margin < MIN_DIGIT_MARGIN:
                failure_reason = f"{spec['source']}:confidence"
                break

            if value is None:
                failure_reason = f"{spec['source']}:empty"
                break
            if value < spec["min"] or value > spec["max"]:
                failure_reason = f"{spec['source']}:range"
                break

            parsed_stats[str(spec["canonical"])] = value
            digit_scores.append(max_score)
            digit_margins.append(min_margin)

        if failure_reason:
            skipped_parse_failed += 1
            failure_counts[failure_reason] += 1
            continue

        labels["agent_stats"] = parsed_stats
        _update_snapshot(labels, parsed_stats)
        _update_review_final(labels, args.reviewer_id)
        applied += 1
        locale_counts[str(record.get("locale") or "UNKNOWN")] += 1
        session_counts[str(record.get("sessionId") or "")] += 1

        if len(examples) < 12:
            examples.append(
                {
                    "recordId": str(record.get("id") or ""),
                    "focusAgentId": str(record.get("focusAgentId") or ""),
                    "locale": str(record.get("locale") or ""),
                    "maxDigitScore": round(max(digit_scores), 4),
                    "minDigitMargin": round(min(digit_margins), 4),
                    "stats": parsed_stats,
                    "path": path_value,
                }
            )

    report = {
        "generatedAt": utc_now(),
        "workflow": args.workflow,
        "reviewerId": args.reviewer_id,
        "processedRecords": processed,
        "appliedRecords": applied,
        "skippedExisting": skipped_existing,
        "skippedMissingAssets": skipped_missing_assets,
        "skippedNonReviewed": skipped_nonreviewed,
        "skippedDerived": skipped_derived,
        "skippedParseFailed": skipped_parse_failed,
        "digitTemplateCount": len(templates),
        "maxDigitScore": MAX_DIGIT_SCORE,
        "minDigitMargin": MIN_DIGIT_MARGIN,
        "localeCounts": dict(sorted(locale_counts.items())),
        "sessionCounts": dict(sorted(session_counts.items())),
        "failureCounts": dict(sorted(failure_counts.items())),
        "examples": examples,
    }

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
