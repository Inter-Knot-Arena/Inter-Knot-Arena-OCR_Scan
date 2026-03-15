from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

_UID_WIDGET_BOX = (0.02, 0.095, 0.19, 0.062)


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reviewed_parent_uid_panels(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        labels = record.get("labels")
        if not isinstance(labels, dict):
            continue
        if str(record.get("qaStatus") or "").strip().lower() != "reviewed":
            continue
        if not isinstance(labels.get("reviewFinal"), dict):
            continue
        role = str((record.get("source") or {}).get("screenRole") or record.get("screenRole") or "").strip()
        if role != "uid_panel":
            continue
        if str(record.get("kind") or "").strip() == "derived_uid_digit":
            continue
        output.append(record)
    return output


def _normalize_uid(labels: Dict[str, Any]) -> str:
    for key in ("uid_full", "label"):
        value = labels.get(key)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            if 6 <= len(digits) <= 12:
                return digits
    payload = labels.get("uid")
    if isinstance(payload, dict):
        value = payload.get("value")
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            if 6 <= len(digits) <= 12:
                return digits
    return ""


def _load_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _fractional_crop(image: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    x, y, w, h = box
    x0 = max(0, min(int(round(width * x)), width - 1))
    y0 = max(0, min(int(round(height * y)), height - 1))
    x1 = max(x0 + 1, min(int(round(width * (x + w))), width))
    y1 = max(y0 + 1, min(int(round(height * (y + h))), height))
    crop = image[y0:y1, x0:x1]
    return crop if crop.size else None


def _sample_group(records: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    if len(records) <= sample_size:
        return records
    indexes = np.linspace(0, len(records) - 1, sample_size, dtype=int)
    return [records[int(index)] for index in indexes]


def _thumbnail(image: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((max_height, max_width, 3), 245, dtype=np.uint8)
    offset_x = (max_width - resized_width) // 2
    offset_y = (max_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def _uid_widget_thumbnail(image: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    crop = _fractional_crop(image, _UID_WIDGET_BOX)
    if crop is None:
        crop = image
    return _thumbnail(crop, max_width=max_width, max_height=max_height)


def _draw_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (255, 255, 255), thickness=-1)
    cv2.putText(
        output,
        text[:80],
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return output


def _contact_sheet(records: List[Dict[str, Any]], output_path: Path, current_uid: str) -> str:
    tile_width = 520
    tile_height = 140
    padding = 12
    title_height = 40
    thumbs: List[np.ndarray] = []

    for record in records:
        path = Path(str(record.get("path") or ""))
        image = _load_image(path)
        if image is None:
            image = np.full((tile_height, tile_width, 3), 220, dtype=np.uint8)
            label = f"missing: {path.name}"
        else:
            image = _uid_widget_thumbnail(image, max_width=tile_width, max_height=tile_height)
            label = path.name
        thumbs.append(_draw_label(image, label))

    if not thumbs:
        return ""

    columns = min(3, len(thumbs))
    rows = int(math.ceil(len(thumbs) / columns))
    sheet_width = padding + columns * (tile_width + padding)
    sheet_height = title_height + padding + rows * (tile_height + padding)
    sheet = np.full((sheet_height, sheet_width, 3), 250, dtype=np.uint8)

    title = f"current_uid={current_uid or 'missing'} samples={len(records)}"
    cv2.putText(sheet, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2, cv2.LINE_AA)

    for index, thumb in enumerate(thumbs):
        row = index // columns
        column = index % columns
        x = padding + column * (tile_width + padding)
        y = title_height + padding + row * (tile_height + padding)
        sheet[y : y + tile_height, x : x + tile_width] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return str(output_path)


def _group_key(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    uid = _normalize_uid(labels)
    return uid or "__missing_uid__"


def _session_counts(records: Iterable[Dict[str, Any]]) -> str:
    counts = Counter(str(record.get("sessionId") or "") for record in records)
    return "; ".join(f"{session}:{count}" for session, count in sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build grouped review pack for reviewed uid_panel records.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sheet-dir", required=True)
    parser.add_argument("--samples-per-group", type=int, default=6)
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest).resolve())
    records = _reviewed_parent_uid_panels(manifest)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_group_key(record)].append(record)

    output_rows: List[Dict[str, str]] = []
    sheet_dir = Path(args.sheet_dir).resolve()
    for group_index, (current_uid, group_records) in enumerate(sorted(grouped.items()), start=1):
        sorted_records = sorted(group_records, key=lambda item: (str(item.get("sessionId") or ""), str(item.get("path") or "")))
        sampled_records = _sample_group(sorted_records, max(1, int(args.samples_per_group)))
        sheet_name = f"{group_index:03d}_{current_uid if current_uid != '__missing_uid__' else 'missing'}.jpg"
        sheet_path = _contact_sheet(sampled_records, sheet_dir / sheet_name, current_uid if current_uid != "__missing_uid__" else "")
        output_rows.append(
            {
                "group_id": f"{group_index:03d}",
                "current_uid_full": "" if current_uid == "__missing_uid__" else current_uid,
                "record_count": str(len(sorted_records)),
                "session_counts": _session_counts(sorted_records),
                "contact_sheet_path": sheet_path,
                "sample_paths_json": json.dumps([str(record.get("path") or "") for record in sampled_records], ensure_ascii=False),
                "check": "",
                "correct_uid_full": "",
                "notes": "",
            }
        )

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "group_id",
                "current_uid_full",
                "record_count",
                "session_counts",
                "contact_sheet_path",
                "sample_paths_json",
                "check",
                "correct_uid_full",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(json.dumps({"groupCount": len(output_rows), "outputCsv": str(output_csv), "sheetDir": str(sheet_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
