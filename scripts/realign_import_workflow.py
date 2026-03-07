from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest
from ocr_dataset_policy import (
    ACCOUNT_IMPORT_WORKFLOW,
    ACCOUNT_UNKNOWN_ROLE,
    COMBAT_REFERENCE_WORKFLOW,
    COMBAT_ROLE,
    apply_record_policy_defaults,
    apply_source_policy_defaults,
    default_role_for_head,
    source_index_from_manifest,
)


def _record_head(record: Dict[str, Any]) -> str:
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def _infer_live_role(heads: List[str]) -> str:
    unique = sorted({head for head in heads if head and head != "unknown"})
    if len(unique) == 1:
        return default_role_for_head(unique[0])
    return ACCOUNT_UNKNOWN_ROLE


def _realign_sources(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sources = manifest.get("sources", [])
    records = manifest.get("records", [])
    if not isinstance(sources, list) or not isinstance(records, list):
        raise ValueError("manifest.sources and manifest.records must be arrays")

    heads_by_source: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        source_id = str(record.get("sourceId") or "").strip()
        if not source_id:
            continue
        heads_by_source[source_id].append(_record_head(record))

    source_index = source_index_from_manifest(sources)
    for source_id, source in source_index.items():
        if source_id.startswith("live_session_"):
            source["workflow"] = ACCOUNT_IMPORT_WORKFLOW
            source["screenRole"] = _infer_live_role(heads_by_source.get(source_id, []))
            source["eligibleHeads"] = sorted({head for head in heads_by_source.get(source_id, []) if head and head != "unknown"})
            source["licenseNote"] = "self-captured account import"
        else:
            source["workflow"] = COMBAT_REFERENCE_WORKFLOW
            source["screenRole"] = COMBAT_ROLE
            source["eligibleHeads"] = []
        apply_source_policy_defaults(source)
    return source_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Realign OCR dataset sources/records to account-import vs combat-reference workflows.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    source_index = _realign_sources(manifest)

    workflow_counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        source_id = str(record.get("sourceId") or "").strip()
        source = source_index.get(source_id, {})
        if source_id.startswith("live_session_"):
            record["workflow"] = ACCOUNT_IMPORT_WORKFLOW
            record["screenRole"] = default_role_for_head(_record_head(record))
            record["eligibleHeads"] = [_record_head(record)]
        else:
            record["workflow"] = COMBAT_REFERENCE_WORKFLOW
            record["screenRole"] = COMBAT_ROLE
            record["eligibleHeads"] = []
        apply_record_policy_defaults(record, source_index)
        workflow_counts[str(record.get("workflow") or "unknown")] += 1
        if bool(record.get("ocrImportEligible")):
            eligible_counts[_record_head(record)] += 1

    save_manifest(manifest_path, manifest)
    summary = {
        "workflowCounts": dict(workflow_counts),
        "eligibleHeadCounts": dict(eligible_counts),
        "sourceWorkflowCounts": dict(Counter(str(source.get("workflow") or "unknown") for source in source_index.values())),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
