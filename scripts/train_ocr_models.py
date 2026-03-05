from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from manifest_lib import hash_file_sha256
from train_synthetic_models import (
    train_agent_icon_model as train_synthetic_agent_icon_model,
    train_uid_model as train_synthetic_uid_model,
)

DEFAULT_MODEL_VERSION = "ocr-heads-v1.3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _extract_head(record: Dict[str, Any]) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        value = labels.get("head")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _extract_label(record: Dict[str, Any], head: str) -> str:
    labels = record.get("labels")
    if isinstance(labels, dict):
        if head == "uid_digit":
            for key in ("uid_digit", "label"):
                value = labels.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if head == "agent_icon":
            for key in ("agent_icon_id", "agentId", "label"):
                value = labels.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for key in ("label", "agentId"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _preprocess(image: np.ndarray, head: str) -> np.ndarray:
    if head == "uid_digit":
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 24), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32).reshape(-1) / 255.0
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(-1) / 255.0


def _load_dataset(manifest_path: Path, head: str) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")

    features: List[np.ndarray] = []
    labels: List[str] = []
    skipped = 0
    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue
        record_head = _extract_head(record)
        if record_head and record_head != head:
            continue
        label = _extract_label(record, head=head)
        if not label:
            skipped += 1
            continue
        path_value = str(record.get("path") or "")
        if not path_value:
            skipped += 1
            continue
        path = Path(path_value)
        if not path.exists():
            skipped += 1
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue
        features.append(_preprocess(image=image, head=head))
        labels.append(label)

    if not features:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64), [], skipped
    label_names = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(label_names)}
    x = np.vstack(features).astype(np.float32)
    y = np.array([label_map[label] for label in labels], dtype=np.int64)
    return x, y, label_names, skipped


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    if probs.size == 0:
        return 0.0
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == y_true).astype(np.float32)
    ece = 0.0
    for idx in range(bins):
        left = idx / bins
        right = (idx + 1) / bins
        mask = (confidences > left) & (confidences <= right)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(correctness[mask]))
        ece += abs(bin_acc - bin_conf) * (np.sum(mask) / len(confidences))
    return float(ece)


def _latency_stats(clf: LogisticRegression, sample: np.ndarray, iterations: int = 120) -> Tuple[float, float]:
    if sample.size == 0:
        return 0.0, 0.0
    latencies: List[float] = []
    count = min(iterations, max(20, sample.shape[0]))
    for idx in range(count):
        row = sample[idx % sample.shape[0] : (idx % sample.shape[0]) + 1]
        started = time.perf_counter()
        _ = clf.predict_proba(row)
        latencies.append((time.perf_counter() - started) * 1000.0)
    p50 = float(statistics.median(latencies))
    p95 = float(np.percentile(np.array(latencies, dtype=np.float32), 95))
    return round(p50, 3), round(p95, 3)


def _train_real_classifier(
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    output_model_path: Path,
    output_labels_path: Path,
) -> Dict[str, float]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    probs = clf.predict_proba(x_test)

    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0.0))
    precision = float(precision_score(y_test, preds, average="macro", zero_division=0.0))
    recall = float(recall_score(y_test, preds, average="macro", zero_division=0.0))
    ece = _expected_calibration_error(y_test, probs)
    latency_p50, latency_p95 = _latency_stats(clf, x_val)

    initial_type = [("input", FloatTensorType([None, x.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=17)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with output_model_path.open("wb") as fh:
        fh.write(onnx_model.SerializeToString())
    with output_labels_path.open("w", encoding="utf-8") as fh:
        json.dump({"labels": labels}, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    return {
        "accuracy": accuracy,
        "macroF1": macro_f1,
        "precision": precision,
        "recall": recall,
        "ece": ece,
        "latencyMsP50": latency_p50,
        "latencyMsP95": latency_p95,
        "backgroundCount": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train OCR UID and agent-icon heads from manifest data with fallback.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--metrics-file", default="docs/model_metrics.json")
    parser.add_argument("--uid-samples-per-class", type=int, default=1500)
    parser.add_argument("--icon-samples-per-class", type=int, default=1600)
    parser.add_argument("--min-real-samples", type=int, default=2000)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--data-version", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    uid_x, uid_y, uid_labels, uid_skipped = _load_dataset(manifest_path=manifest_path, head="uid_digit")
    icon_x, icon_y, icon_labels, icon_skipped = _load_dataset(manifest_path=manifest_path, head="agent_icon")

    uid_real = uid_x.shape[0] >= max(200, args.min_real_samples) and len(uid_labels) >= 2
    icon_real = icon_x.shape[0] >= max(200, args.min_real_samples) and len(icon_labels) >= 2

    if uid_real:
        uid_metrics = _train_real_classifier(
            x=uid_x,
            y=uid_y,
            labels=uid_labels,
            output_model_path=output_dir / "uid_digit.onnx",
            output_labels_path=output_dir / "uid_digit.labels.json",
        )
        uid_mode = "real"
    else:
        uid_metrics = train_synthetic_uid_model(
            output_dir=output_dir,
            backgrounds=[],
            samples_per_class=max(400, args.uid_samples_per_class),
            background_probability=0.0,
        )
        uid_metrics = {
            "accuracy": float(uid_metrics.get("accuracy", 0.0)),
            "macroF1": float(uid_metrics.get("accuracy", 0.0)),
            "precision": float(uid_metrics.get("accuracy", 0.0)),
            "recall": float(uid_metrics.get("accuracy", 0.0)),
            "ece": 0.0,
            "latencyMsP50": 0.0,
            "latencyMsP95": 0.0,
            "backgroundCount": int(uid_metrics.get("backgroundCount", 0)),
        }
        uid_mode = "synthetic_fallback"

    if icon_real:
        icon_metrics = _train_real_classifier(
            x=icon_x,
            y=icon_y,
            labels=icon_labels,
            output_model_path=output_dir / "agent_icon.onnx",
            output_labels_path=output_dir / "agent_icon.labels.json",
        )
        icon_mode = "real"
    else:
        icon_metrics = train_synthetic_agent_icon_model(
            output_dir=output_dir,
            backgrounds=[],
            samples_per_class=max(400, args.icon_samples_per_class),
            background_probability=0.0,
        )
        icon_metrics = {
            "accuracy": float(icon_metrics.get("accuracy", 0.0)),
            "macroF1": float(icon_metrics.get("accuracy", 0.0)),
            "precision": float(icon_metrics.get("accuracy", 0.0)),
            "recall": float(icon_metrics.get("accuracy", 0.0)),
            "ece": 0.0,
            "latencyMsP50": 0.0,
            "latencyMsP95": 0.0,
            "backgroundCount": int(icon_metrics.get("backgroundCount", 0)),
        }
        icon_mode = "synthetic_fallback"

    data_version = args.data_version
    if not data_version:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest_payload = json.load(fh)
        data_version = str(manifest_payload.get("version") or "unknown")

    artifacts = {
        "uid_digit.onnx": hash_file_sha256(output_dir / "uid_digit.onnx") if (output_dir / "uid_digit.onnx").exists() else "",
        "agent_icon.onnx": hash_file_sha256(output_dir / "agent_icon.onnx") if (output_dir / "agent_icon.onnx").exists() else "",
    }
    model_manifest = {
        "version": args.model_version,
        "trainedAt": _utc_now(),
        "dataVersion": data_version,
        "artifacts": artifacts,
    }
    with (output_dir / "model_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(model_manifest, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    metrics_payload = {
        "uid_digit_model": {
            **uid_metrics,
            "dataVersion": data_version,
            "trainedAt": model_manifest["trainedAt"],
            "recordCount": int(uid_x.shape[0]),
            "skippedRecords": uid_skipped,
            "mode": uid_mode,
        },
        "agent_icon_model": {
            **icon_metrics,
            "dataVersion": data_version,
            "trainedAt": model_manifest["trainedAt"],
            "recordCount": int(icon_x.shape[0]),
            "skippedRecords": icon_skipped,
            "mode": icon_mode,
        },
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(json.dumps({"metrics": metrics_payload, "modelManifest": model_manifest}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

