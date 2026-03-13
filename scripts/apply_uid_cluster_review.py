from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_dataset_policy import ACCOUNT_IMPORT_WORKFLOW, UID_PANEL_ROLE


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_labels(record: Dict[str, Any]) -> Dict[str, Any]:
    labels = record.get("labels")
    if isinstance(labels, dict):
        return labels
    labels = {}
    record["labels"] = labels
    return labels


def _normalize_uid(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str):
        return ""
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits if 6 <= len(digits) <= 12 else ""


def _existing_uid(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    if not isinstance(labels, dict):
        return ""
    for key in ("uid_full", "label", "uid"):
        normalized = _normalize_uid(labels.get(key))
        if normalized:
            return normalized
    return ""


def _cluster_uid_mapping(payload: Any) -> Dict[int, str]:
    if isinstance(payload, dict):
        source = payload.get("clusterUidValues", payload)
    else:
        source = payload
    if not isinstance(source, dict):
        raise ValueError("mapping payload must be an object")

    output: Dict[int, str] = {}
    for raw_cluster_id, raw_uid in source.items():
        try:
            cluster_id = int(str(raw_cluster_id).strip())
        except ValueError as exc:
            raise ValueError(f"Invalid cluster id in mapping: {raw_cluster_id}") from exc
        uid_full = _normalize_uid(raw_uid)
        if not uid_full:
            raise ValueError(f"Invalid UID value for cluster {cluster_id}: {raw_uid}")
        output[cluster_id] = uid_full
    return output


def _cluster_index(clusters: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    output: Dict[int, Dict[str, Any]] = {}
    for entry in clusters:
        if not isinstance(entry, dict):
            continue
        raw_cluster_id = entry.get("clusterId")
        try:
            cluster_id = int(raw_cluster_id)
        except (TypeError, ValueError):
            continue
        output[cluster_id] = entry
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply reviewed full-UID truth to clustered uid_panel records.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--cluster-report", required=True)
    parser.add_argument("--mapping-json", required=True)
    parser.add_argument("--reviewer-id", default="manual_uid_cluster_review")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    cluster_report_path = Path(args.cluster_report).resolve()
    mapping_path = Path(args.mapping_json).resolve()

    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    cluster_payload = _load_json(cluster_report_path)
    if not isinstance(cluster_payload, list):
        raise ValueError("cluster report must be a JSON array")
    clusters = _cluster_index(cluster_payload)
    mapping = _cluster_uid_mapping(_load_json(mapping_path))

    record_index = {
        str(record.get("id") or ""): record
        for record in records
        if isinstance(record, dict) and str(record.get("id") or "")
    }

    updated_records = 0
    updated_clusters = 0
    skipped_missing_cluster: List[int] = []
    skipped_conflicts: List[Dict[str, Any]] = []
    applied_cluster_sizes: Dict[str, int] = {}

    for cluster_id, uid_full in sorted(mapping.items()):
        cluster = clusters.get(cluster_id)
        if not isinstance(cluster, dict):
            skipped_missing_cluster.append(cluster_id)
            continue

        member_ids = cluster.get("memberIds")
        if not isinstance(member_ids, list):
            skipped_missing_cluster.append(cluster_id)
            continue

        cluster_updates = 0
        representative_path = str(cluster.get("representativePath") or "")
        for member_id in member_ids:
            record = record_index.get(str(member_id))
            if not isinstance(record, dict):
                continue
            if str(record.get("workflow") or "").strip() != ACCOUNT_IMPORT_WORKFLOW:
                continue
            if str(record.get("screenRole") or "").strip() != UID_PANEL_ROLE:
                continue

            existing_uid = _existing_uid(record)
            if existing_uid and existing_uid != uid_full:
                skipped_conflicts.append(
                    {
                        "clusterId": cluster_id,
                        "recordId": str(record.get("id") or ""),
                        "existingUid": existing_uid,
                        "requestedUid": uid_full,
                    }
                )
                continue

            labels = _record_labels(record)
            labels["uid_full"] = uid_full
            labels["uid"] = {"value": uid_full}
            labels["label"] = uid_full
            labels["head"] = "uid_digit"
            labels["reviewFinal"] = {
                "reviewer": str(args.reviewer_id),
                "reviewedAt": utc_now(),
                "notes": (
                    f"UID reviewed from cluster {cluster_id} representative; "
                    f"representativePath={representative_path}"
                ),
            }
            record["head"] = "uid_digit"
            record["qaStatus"] = "reviewed"
            cluster_updates += 1
            updated_records += 1

        if cluster_updates:
            updated_clusters += 1
            applied_cluster_sizes[str(cluster_id)] = cluster_updates

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    payload = {
        "generatedAt": utc_now(),
        "manifest": str(manifest_path),
        "clusterReport": str(cluster_report_path),
        "mappingJson": str(mapping_path),
        "reviewerId": str(args.reviewer_id),
        "updatedClusters": updated_clusters,
        "updatedRecords": updated_records,
        "appliedClusterSizes": applied_cluster_sizes,
        "skippedMissingClusterIds": skipped_missing_cluster,
        "skippedConflicts": skipped_conflicts,
        "dryRun": bool(args.dry_run),
    }

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
