from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def default_manifest() -> Dict[str, Any]:
    return {
        "name": "ika_ocr_private_dataset",
        "version": "1.0.0",
        "storagePolicy": {
            "rawDataInGit": False,
            "notes": "Raw data stays in private local storage only.",
        },
        "splits": {"train": [], "val": [], "test": []},
        "sources": [],
        "records": [],
    }


def ensure_dirs(root: Path) -> None:
    for folder in ("raw", "frames", "labels_uid", "labels_agents", "labels_equipment"):
        (root / folder).mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap private OCR dataset directories and manifest.")
    parser.add_argument("--storage-root", required=True, help="Private local storage root path.")
    parser.add_argument("--manifest", default="dataset_manifest.json", help="Manifest path in repo.")
    args = parser.parse_args()

    storage_root = Path(args.storage_root).expanduser().resolve()
    ensure_dirs(storage_root)

    manifest_path = Path(args.manifest).resolve()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = default_manifest()

    payload["privateStorageRoot"] = str(storage_root)
    payload["directoryLayout"] = {
        "raw": str(storage_root / "raw"),
        "frames": str(storage_root / "frames"),
        "labels_uid": str(storage_root / "labels_uid"),
        "labels_agents": str(storage_root / "labels_agents"),
        "labels_equipment": str(storage_root / "labels_equipment"),
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(f"Dataset bootstrap complete. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
