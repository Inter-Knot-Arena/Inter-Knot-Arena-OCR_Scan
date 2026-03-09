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
    parser.add_argument("--agent", dest="agents", action="append", required=True)
    parser.add_argument("--role-order", default=",".join(DEFAULT_ROLE_ORDER))
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--copy-mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--summary-file", default="")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    files = _collect_files(input_root, int(args.start_index), int(args.end_index))
    agents = _canonical_agents(args.agents)
    role_order = _parse_roles(args.role_order)
    skip_map = _parse_skips(args.skip)

    expected_count = len(agents) * len(role_order)
    if len(files) != expected_count:
        raise ValueError(
            f"ordered batch size mismatch: found {len(files)} files for {len(agents)} agents x {len(role_order)} roles = {expected_count}"
        )

    mapping: List[dict] = []
    write_op = shutil.move if args.copy_mode == "move" else shutil.copy2

    for agent_index, agent_id in enumerate(agents):
        agent_dir = output_root / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        start = agent_index * len(role_order)
        group_files = files[start : start + len(role_order)]
        skipped_roles = skip_map.get(agent_id, set())
        record = {
            "agentId": agent_id,
            "sourceFiles": [str(path) for path in group_files],
            "assignedRoles": [],
            "skippedRoles": sorted(skipped_roles),
        }
        for role, source_path in zip(role_order, group_files):
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
