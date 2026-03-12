from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2

from manifest_lib import ensure_manifest_defaults, hash_file_sha256, load_manifest, save_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from locale_utils import resolve_record_locale
from ocr_dataset_policy import (
    ACCOUNT_IMPORT_WORKFLOW,
    AGENT_DETAIL_ROLE,
    AMPLIFIER_DETAIL_ROLE,
    DISK_DETAIL_ROLE,
    EQUIPMENT_ROLE,
    MINDSCAPE_ROLE,
    ROSTER_ROLE,
    UID_PANEL_ROLE,
    apply_record_policy_defaults,
    apply_source_policy_defaults,
    source_index_from_manifest,
)
from roster_taxonomy import canonicalize_agent_label


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

ROLE_TO_HEAD = {
    UID_PANEL_ROLE: "uid_digit",
    ROSTER_ROLE: "agent_icon",
    AGENT_DETAIL_ROLE: "agent_icon",
    EQUIPMENT_ROLE: "equipment",
    DISK_DETAIL_ROLE: "equipment",
    AMPLIFIER_DETAIL_ROLE: "equipment",
    MINDSCAPE_ROLE: "equipment",
}


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        yield path


def _normalized_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = [part.lower() for part in relative.parts]
    return "/".join(parts)


def _extract_slot(text: str) -> int | None:
    match = re.search(r"\b(?:disk|disc)[_\-\s]*([1-6])\b", text)
    if not match:
        return None
    return int(match.group(1))


def _detect_focus_agent(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    for part in relative.parts[:-1]:
        token = str(part).strip().lower()
        if not token:
            continue
        canonical = canonicalize_agent_label(token, include_upcoming=True)
        if canonical:
            return canonical
    return ""


def _detect_screen_role(path: Path, root: Path) -> tuple[str, Dict[str, Any]] | None:
    normalized = _normalized_name(path, root)

    if re.search(r"\buid\b", normalized) or "profile" in normalized:
        return UID_PANEL_ROLE, {"screenAlias": "uid"}
    if "roster" in normalized or "agent_menu" in normalized or "agents_menu" in normalized:
        return ROSTER_ROLE, {"screenAlias": "roster"}
    if "mindscape" in normalized or "cinema" in normalized or re.search(r"\bmind\b", normalized):
        return MINDSCAPE_ROLE, {"screenAlias": "mindscape"}
    if "amplificator" in normalized or "amplifier" in normalized or "w_engine" in normalized or "wengine" in normalized:
        return AMPLIFIER_DETAIL_ROLE, {"screenAlias": "amplificator"}

    slot_index = _extract_slot(normalized)
    if slot_index is not None:
        return DISK_DETAIL_ROLE, {"screenAlias": f"disk{slot_index}", "slotIndex": slot_index}

    if "agent_detail" in normalized or "characteristics" in normalized or "stats" in normalized:
        return AGENT_DETAIL_ROLE, {"screenAlias": "agent_detail"}
    if re.search(r"\bequipment\b", normalized) or "loadout" in normalized or "gear" in normalized:
        return EQUIPMENT_ROLE, {"screenAlias": "equipment"}
    return None


def _source_tags(role: str, hints: Dict[str, Any]) -> list[str]:
    if role == UID_PANEL_ROLE:
        return ["account", "profile", "uid"]
    if role == ROSTER_ROLE:
        return ["account", "roster", "agents"]
    if role == AGENT_DETAIL_ROLE:
        return ["account", "agent_detail", "agent", "stats"]
    if role == EQUIPMENT_ROLE:
        return ["account", "equipment", "inventory"]
    if role == DISK_DETAIL_ROLE:
        slot_index = hints.get("slotIndex")
        tags = ["account", "equipment", "inventory", "disc_detail", "disc"]
        if isinstance(slot_index, int):
            tags.append(f"slot_{slot_index}")
        return tags
    if role == AMPLIFIER_DETAIL_ROLE:
        return ["account", "equipment", "inventory", "amplifier"]
    if role == MINDSCAPE_ROLE:
        return ["account", "mindscape", "cinema"]
    return ["account"]


def _read_dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return 0, 0
    height, width = image.shape[:2]
    return int(width), int(height)


def _guess_resolution(width: int, height: int) -> str:
    if width >= 2560 or height >= 1440:
        return "1440p"
    if width >= 1920 or height >= 1080:
        return "1080p"
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "unknown"


def _copy_target(raw_dir: Path, batch_id: str, role: str, source_id: str, original: Path) -> Path:
    target_dir = raw_dir / "manual_screens" / batch_id / role
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{source_id}{original.suffix.lower()}"


def _existing_sha256(records: list[Dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        digest = str(record.get("sha256") or "").strip().lower()
        if digest:
            values.add(digest)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Import manually captured account screenshots into the OCR manifest.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--input-root", required=True, help="Folder with uid/roster/agent_detail/equipment/disk1-6/amplificator/mindscape screenshots.")
    parser.add_argument("--batch-id", default="", help="Stable import batch id. Defaults to timestamp-based id.")
    parser.add_argument(
        "--locale",
        default="RU",
        help="Locale for imported records. Use EN or RU for single-locale batches, or AUTO/MIXED to resolve per file from path hints.",
    )
    parser.add_argument(
        "--fallback-locale",
        default="UNKNOWN",
        help="Fallback locale when --locale AUTO/MIXED cannot resolve a file from path hints.",
    )
    parser.add_argument("--resolution", default="", help="Override resolution label. Default is inferred from image size.")
    parser.add_argument("--collector", default="")
    parser.add_argument("--game-patch", default="unknown")
    parser.add_argument("--capture-date", default="", help="Optional ISO date/time for imported screenshots.")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    layout = manifest.get("directoryLayout", {})
    if not isinstance(layout, dict):
        raise ValueError("manifest.directoryLayout must be an object")
    raw_folder = str(layout.get("raw") or "").strip()
    if not raw_folder:
        raise ValueError("raw directory is not configured; run bootstrap_dataset.py first")

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise ValueError(f"input root not found: {input_root}")

    raw_dir = Path(raw_folder).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    batch_id = str(args.batch_id or f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}").strip()
    collector = str(args.collector or Path.home().name).strip() or "unknown"
    explicit_capture_date = str(args.capture_date).strip()

    sources = manifest.setdefault("sources", [])
    records = manifest.setdefault("records", [])
    existing_hashes = _existing_sha256(records)

    imported = 0
    skipped_duplicate = 0
    skipped_unclassified = 0
    role_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    skipped_files: list[str] = []

    for image_path in _iter_images(input_root):
        detection = _detect_screen_role(image_path, input_root)
        if detection is None:
            skipped_unclassified += 1
            skipped_files.append(str(image_path))
            continue

        role, capture_hints = detection
        focus_agent_id = _detect_focus_agent(image_path, input_root)
        sha256 = hash_file_sha256(image_path)
        if sha256.lower() in existing_hashes:
            skipped_duplicate += 1
            continue

        width, height = _read_dimensions(image_path)
        resolution = str(args.resolution).strip() or _guess_resolution(width, height)
        capture_date = explicit_capture_date or _iso_from_timestamp(image_path.stat().st_mtime)
        head = ROLE_TO_HEAD[role]
        source_id = f"{batch_id}_{role}_{sha256[:12]}"
        record_id = f"{source_id}_record"
        target_path = _copy_target(raw_dir, batch_id, role, source_id, image_path)
        record_locale = resolve_record_locale(
            image_path,
            input_root,
            str(args.locale),
            fallback=str(args.fallback_locale),
        )

        source = {
            "sourceId": source_id,
            "url": "",
            "captureDate": capture_date,
            "licenseNote": "self-captured account import screenshot",
            "locale": record_locale,
            "resolution": resolution,
            "gamePatch": str(args.game_patch).strip() or "unknown",
            "collector": collector,
            "sourceType": "manual-screenshot",
            "workflow": ACCOUNT_IMPORT_WORKFLOW,
            "screenRole": role,
            "sourceTags": _source_tags(role, capture_hints),
            "originPath": str(image_path),
        }
        if focus_agent_id:
            source["focusAgentId"] = focus_agent_id
        sources.append(apply_source_policy_defaults(source))

        if not args.dry_run:
            shutil.copy2(image_path, target_path)

        source_index = source_index_from_manifest(sources)
        record = {
            "id": record_id,
            "sourceId": source_id,
            "sessionId": batch_id,
            "matchId": "",
            "kind": "manual_screenshot",
            "head": head,
            "state": "other",
            "locale": record_locale,
            "resolution": resolution,
            "path": str(target_path.resolve()),
            "labelsPath": "",
            "qaStatus": "unlabeled",
            "labels": {},
            "suggestedLabels": {},
            "sha256": sha256,
            "originalFileName": image_path.name,
            "dimensions": {"width": width, "height": height},
            "captureHints": capture_hints,
        }
        if focus_agent_id:
            record["focusAgentId"] = focus_agent_id
        records.append(apply_record_policy_defaults(record, source_index))
        existing_hashes.add(sha256.lower())
        imported += 1
        role_counts[role] += 1
        locale_counts[record_locale] += 1

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    result = {
        "batchId": batch_id,
        "inputRoot": str(input_root),
        "imported": imported,
        "skippedDuplicate": skipped_duplicate,
        "skippedUnclassified": skipped_unclassified,
        "roleCounts": dict(role_counts),
        "localeCounts": dict(locale_counts),
        "unclassifiedFiles": skipped_files[:50],
        "dryRun": bool(args.dry_run),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
