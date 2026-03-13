from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from manifest_lib import ensure_manifest_defaults, load_manifest, save_manifest


def _read_dimensions(path: Path) -> tuple[int, int]:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return 0, 0
    if buffer.size <= 0:
        return 0, 0
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
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


def _source_index(sources: list[Dict[str, Any]]) -> dict[str, Dict[str, Any]]:
    output: dict[str, Dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("sourceId") or "").strip()
        if source_id:
            output[source_id] = source
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill resolution and dimensions for imported manual screenshots.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--session-id", action="append", dest="session_ids", default=[])
    parser.add_argument("--workflow", default="account_import")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = ensure_manifest_defaults(load_manifest(manifest_path))
    sources = manifest.get("sources", [])
    records = manifest.get("records", [])
    if not isinstance(sources, list) or not isinstance(records, list):
        raise ValueError("manifest sources/records must be arrays")

    session_filter = {str(value).strip() for value in args.session_ids if str(value).strip()}
    source_by_id = _source_index(sources)

    updated_records = 0
    updated_sources = 0
    missing_paths = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("kind") or "") != "manual_screenshot":
            continue
        if str(record.get("workflow") or "") != str(args.workflow):
            continue
        session_id = str(record.get("sessionId") or "")
        if session_filter and session_id not in session_filter:
            continue

        source_id = str(record.get("sourceId") or "")
        source = source_by_id.get(source_id)
        if not source:
            continue

        record_path = Path(str(record.get("path") or "")).expanduser()
        if not record_path.exists():
            fallback = Path(str(source.get("originPath") or "")).expanduser()
            if fallback.exists():
                record_path = fallback
            else:
                missing_paths += 1
                continue

        width, height = _read_dimensions(record_path)
        resolution = _guess_resolution(width, height)

        dimensions = record.get("dimensions")
        if not isinstance(dimensions, dict):
            dimensions = {}
            record["dimensions"] = dimensions

        record_changed = False
        if int(dimensions.get("width") or 0) != width:
            dimensions["width"] = int(width)
            record_changed = True
        if int(dimensions.get("height") or 0) != height:
            dimensions["height"] = int(height)
            record_changed = True
        if str(record.get("resolution") or "") != resolution:
            record["resolution"] = resolution
            record_changed = True
        if record_changed:
            updated_records += 1

        if str(source.get("resolution") or "") != resolution:
            source["resolution"] = resolution
            updated_sources += 1

    summary = {
        "workflow": str(args.workflow),
        "sessionIds": sorted(session_filter),
        "updatedRecords": updated_records,
        "updatedSources": updated_sources,
        "missingPaths": missing_paths,
        "dryRun": bool(args.dry_run),
    }

    if not args.dry_run:
        save_manifest(manifest_path, manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
