from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Set

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roster_taxonomy import canonicalize_agent_label


DEFAULT_ROLE_ORDER = [
    "agent_detail",
    "equipment",
    "disk1",
    "disk2",
    "disk3",
    "disk4",
    "disk5",
    "disk6",
    "amplificator",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _index_from_name(path: Path) -> int | None:
    match = re.search(r"\((\d+)\)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _collect_files(root: Path, start_index: int, end_index: int) -> List[Path]:
    output: List[Path] = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        index = _index_from_name(path)
        if index is None:
            continue
        if start_index <= index <= end_index:
            output.append(path)
    output.sort(key=lambda item: (_index_from_name(item) or 0, item.name.lower()))
    return output


def _parse_roles(raw: str) -> List[str]:
    output: List[str] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if token:
            output.append(token)
    return output or list(DEFAULT_ROLE_ORDER)


def _parse_agent_packs(values: Sequence[str]) -> List[dict]:
    output: List[dict] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"agent pack must be agent_label:role1,role2 -> {text!r}")
        agent_token, roles_text = text.split(":", 1)
        canonical = canonicalize_agent_label(agent_token.strip(), include_upcoming=True)
        if not canonical:
            raise ValueError(f"invalid agent in agent pack: {agent_token!r}")
        roles = _parse_roles(roles_text)
        if not roles:
            raise ValueError(f"agent pack has no roles: {text!r}")
        output.append({"agentId": canonical, "roles": roles})
    return output


def _parse_skips(values: Sequence[str]) -> Dict[str, Set[str]]:
    output: Dict[str, Set[str]] = {}
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"skip rule must be canonical_agent_id:role1,role2 -> {text!r}")
        agent_token, roles_text = text.split(":", 1)
        canonical_agent = canonicalize_agent_label(agent_token.strip(), include_upcoming=True)
        if not canonical_agent:
            raise ValueError(f"invalid agent in skip rule: {agent_token!r}")
        roles = {role.strip() for role in roles_text.split(",") if role.strip()}
        if not roles:
            raise ValueError(f"skip rule has no roles: {text!r}")
        output.setdefault(canonical_agent, set()).update(roles)
    return output


def _canonical_agents(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    for raw in values:
        canonical = canonicalize_agent_label(raw, include_upcoming=True)
        if not canonical:
            raise ValueError(f"invalid agent value: {raw!r}")
        output.append(canonical)
    if not output:
        raise ValueError("at least one --agent value is required")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize ordered generic screenshots into per-agent OCR drop folders.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int)
    parser.add_argument("--agent", dest="agents", action="append", default=[])
    parser.add_argument("--agent-pack", action="append", default=[])
    parser.add_argument("--role-order", default=",".join(DEFAULT_ROLE_ORDER))
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--copy-mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--summary-file", default="")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = _collect_files(input_root, int(args.start_index), int(args.end_index))
    agent_packs = _parse_agent_packs(args.agent_pack)
    skip_map = _parse_skips(args.skip)
    if agent_packs:
        plan = agent_packs
        role_order = []
        agents = [entry["agentId"] for entry in plan]
        expected_count = sum(len(entry["roles"]) for entry in plan)
        if args.agents:
            raise ValueError("--agent and --agent-pack cannot be combined")
    else:
        if not args.agents:
            raise ValueError("at least one --agent or --agent-pack value is required")
        agents = _canonical_agents(args.agents)
        role_order = _parse_roles(args.role_order)
        plan = [{"agentId": agent_id, "roles": role_order} for agent_id in agents]
        expected_count = len(agents) * len(role_order)

    if len(files) != expected_count:
        raise ValueError(
            f"ordered batch size mismatch: found {len(files)} files for planned role count {expected_count}"
        )

    mapping: List[dict] = []
    write_op = shutil.move if args.copy_mode == "move" else shutil.copy2

    cursor = 0
    for pack in plan:
        agent_id = str(pack["agentId"])
        pack_roles = list(pack["roles"])
        agent_dir = output_root / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        group_files = files[cursor : cursor + len(pack_roles)]
        cursor += len(pack_roles)
        skipped_roles = skip_map.get(agent_id, set())
        record = {
            "agentId": agent_id,
            "sourceFiles": [str(path) for path in group_files],
            "roleOrder": pack_roles,
            "assignedRoles": [],
            "skippedRoles": sorted(skipped_roles),
        }
        for role, source_path in zip(pack_roles, group_files):
            if role in skipped_roles:
                continue
            target_path = agent_dir / f"{role}{source_path.suffix.lower()}"
            write_op(source_path, target_path)
            record["assignedRoles"].append(
                {
                    "role": role,
                    "sourcePath": str(source_path),
                    "targetPath": str(target_path),
                }
            )
        mapping.append(record)

    summary = {
        "inputRoot": str(input_root),
        "outputRoot": str(output_root),
        "startIndex": int(args.start_index),
        "endIndex": int(args.end_index),
        "roleOrder": role_order,
        "agents": agents,
        "agentPacks": plan if agent_packs else [],
        "skipRules": {key: sorted(value) for key, value in sorted(skip_map.items())},
        "materializedAgents": mapping,
    }

    if args.summary_file:
        summary_path = Path(args.summary_file).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
