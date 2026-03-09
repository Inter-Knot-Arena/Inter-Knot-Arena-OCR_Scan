from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest, utc_now

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import canonicalize_agent_label, current_agent_ids


def _normalize_owned_agents(values: Iterable[str]) -> List[str]:
    ordered_current = current_agent_ids()
    seen: set[str] = set()
    owned: List[str] = []
    for raw in values:
        candidate = str(raw or "").strip()
        if not candidate:
            continue
        canonical = canonicalize_agent_label(candidate)
        if not canonical or canonical == "unknown":
            raise ValueError(f"Invalid owned-agent value: {candidate}")
        if canonical not in seen:
            seen.add(canonical)
            owned.append(canonical)
    return [agent for agent in ordered_current if agent in seen]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply confirmed roster ownership truth to imported roster screenshots.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--reviewer-id", default="human")
    parser.add_argument("--notes", default="")
    parser.add_argument("--owned-agent", action="append", dest="owned_agents", default=[])
    args = parser.parse_args()

    owned_agents = _normalize_owned_agents(args.owned_agents)
    if not owned_agents:
        raise ValueError("At least one --owned-agent value is required.")

    current_agents = current_agent_ids()
    not_owned_agents = [agent for agent in current_agents if agent not in owned_agents]

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    applied = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("sessionId") or "") != str(args.session_id):
            continue
        if str(record.get("screenRole") or "") != "roster":
            continue

        labels = record.get("labels")
        if not isinstance(labels, dict):
            labels = {}
            record["labels"] = labels

        labels["head"] = "agent_icon"
        labels["label"] = "roster_ownership_confirmed"
        labels["owned_agent_ids"] = list(owned_agents)
        labels["not_owned_agent_ids"] = list(not_owned_agents)
        labels["owned_agent_count"] = len(owned_agents)
        labels["not_owned_agent_count"] = len(not_owned_agents)
        labels["ownership_confirmed"] = True
        labels["reviewFinal"] = {
            "reviewer": str(args.reviewer_id).strip() or "human",
            "reviewedAt": utc_now(),
            "notes": str(args.notes or "").strip() or "User-confirmed owned roster list.",
        }
        record["qaStatus"] = "reviewed"
        applied += 1

    if applied <= 0:
        raise ValueError(f"No roster records found for sessionId={args.session_id}")

    qa_status = manifest.setdefault("qaStatus", {})
    if isinstance(qa_status, dict):
        qa_status["humanReview"] = "in_progress"
        qa_status["qaUpdatedAt"] = utc_now()

    save_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "sessionId": str(args.session_id),
                "appliedRecords": applied,
                "ownedAgentCount": len(owned_agents),
                "ownedAgentIds": owned_agents,
                "notOwnedAgentCount": len(not_owned_agents),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
