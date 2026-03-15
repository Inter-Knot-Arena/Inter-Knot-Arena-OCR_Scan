from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manifest_lib import hash_file_sha256
from roster_taxonomy import canonicalize_agent_label, current_agent_ids

DEFAULT_MODEL_VERSION = "ocr-heads-v1.4"


@dataclass(slots=True)
class LoadedDataset:
    x: np.ndarray
    y: np.ndarray
    label_names: List[str]
    skipped_records: int
    sample_splits: List[str]
    split_mode: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _torch_cuda_available() -> bool:
    if not _torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


def _cuda_onnx_providers() -> list[str]:
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        available_list = ", ".join(sorted(available)) or "none"
        raise RuntimeError(
            "CUDAExecutionProvider is required for OCR ONNX validation. "
            f"Available providers: {available_list}."
        )
    return ["CUDAExecutionProvider"]


def _normalize_split_name(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in {"train", "val", "test"} else ""


def _build_manifest_split_index(manifest: Dict[str, Any]) -> Dict[str, str]:
    split_index: Dict[str, str] = {}
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        return split_index
    for split_name in ("train", "val", "test"):
        record_ids = splits.get(split_name)
        if not isinstance(record_ids, list):
            continue
        for record_id in record_ids:
            normalized = _normalize_split_name(split_name)
            if normalized:
                split_index[str(record_id)] = normalized
    return split_index


def _select_label_payload(record: Dict[str, Any], label_source: str) -> Dict[str, Any] | None:
    labels = record.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    suggested = record.get("suggestedLabels")
    suggested = suggested if isinstance(suggested, dict) else {}

    is_reviewed = (
        str(record.get("qaStatus") or "").lower() == "reviewed"
        and isinstance(labels.get("reviewFinal"), dict)
    )
    if label_source == "reviewed":
        return labels if is_reviewed else None
    if label_source == "suggested":
        return suggested or None
    if is_reviewed:
        return labels
    if labels:
        return labels
    return suggested or None


def _extract_head(record: Dict[str, Any], payload: Dict[str, Any] | None = None) -> str:
    if isinstance(payload, dict):
        value = payload.get("head")
        if isinstance(value, str) and value.strip():
            return value.strip()
    labels = record.get("labels")
    if isinstance(labels, dict):
        value = labels.get("head")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = record.get("head")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _extract_label(payload: Dict[str, Any], head: str) -> str:
    if head == "uid_digit":
        for key in ("uid_digit", "label"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                digits = "".join(ch for ch in value if ch.isdigit())
                if len(digits) == 1:
                    return digits
        return ""
    if head == "agent_icon":
        for key in ("agent_icon_id", "agentId", "label"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return canonicalize_agent_label(value.strip())
    for key in ("label", "agentId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            if head == "agent_icon":
                return canonicalize_agent_label(value.strip())
            return value.strip()
    return ""


def _preprocess(image: np.ndarray, head: str) -> np.ndarray:
    if head == "uid_digit":
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray > 8
        if np.any(mask):
            ys, xs = np.where(mask)
            gray = gray[int(np.min(ys)) : int(np.max(ys)) + 1, int(np.min(xs)) : int(np.max(xs)) + 1]
        target_width, target_height = 16, 24
        scale = min((target_width - 2) / max(gray.shape[1], 1), (target_height - 2) / max(gray.shape[0], 1))
        resized_width = max(1, int(round(gray.shape[1] * scale)))
        resized_height = max(1, int(round(gray.shape[0] * scale)))
        resized = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
        return canvas.astype(np.float32).reshape(-1) / 255.0
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).reshape(-1) / 255.0


def _load_dataset(manifest_path: Path, head: str, label_source: str, split_source: str) -> LoadedDataset:
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest.records must be an array")
    split_index = _build_manifest_split_index(manifest) if split_source == "manifest" else {}

    features: List[np.ndarray] = []
    labels: List[str] = []
    sample_splits: List[str] = []
    skipped = 0
    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue
        payload = _select_label_payload(record, label_source=label_source)
        if not isinstance(payload, dict):
            skipped += 1
            continue
        record_head = _extract_head(record, payload)
        if record_head and record_head != head:
            continue
        label = _extract_label(payload, head=head)
        if not label:
            skipped += 1
            continue
        split_name = ""
        if split_source == "manifest":
            split_name = _normalize_split_name(split_index.get(str(record.get("id") or "")))
            if not split_name:
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
        sample_splits.append(split_name)

    if not features:
        return LoadedDataset(
            x=np.empty((0, 1), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            label_names=[],
            skipped_records=skipped,
            sample_splits=[],
            split_mode=split_source,
        )
    label_names = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(label_names)}
    x = np.vstack(features).astype(np.float32)
    y = np.array([label_map[label] for label in labels], dtype=np.int64)
    return LoadedDataset(
        x=x,
        y=y,
        label_names=label_names,
        skipped_records=skipped,
        sample_splits=sample_splits,
        split_mode=split_source,
    )


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


def _onnx_latency_stats(model_path: Path, sample: np.ndarray, iterations: int = 120) -> Tuple[float, float]:
    if sample.size == 0 or not model_path.exists():
        return 0.0, 0.0
    session = ort.InferenceSession(str(model_path), providers=_cuda_onnx_providers())
    input_name = session.get_inputs()[0].name
    latencies: List[float] = []
    count = min(iterations, max(20, sample.shape[0]))
    for idx in range(count):
        row = sample[idx % sample.shape[0] : (idx % sample.shape[0]) + 1].astype(np.float32, copy=False)
        started = time.perf_counter()
        _ = session.run(None, {input_name: row})
        latencies.append((time.perf_counter() - started) * 1000.0)
    p50 = float(statistics.median(latencies))
    p95 = float(np.percentile(np.array(latencies, dtype=np.float32), 95))
    return round(p50, 3), round(p95, 3)


def _stratify_target(y: np.ndarray) -> np.ndarray | None:
    if y.size <= 1:
        return None
    values, counts = np.unique(y, return_counts=True)
    if values.size <= 1:
        return None
    if int(np.min(counts)) < 2:
        return None
    return y


def _synthetic_fallback_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": float(raw_metrics.get("accuracy", 0.0)),
        "macroF1": None,
        "precision": None,
        "recall": None,
        "ece": None,
        "latencyMsP50": None,
        "latencyMsP95": None,
        "backgroundCount": int(raw_metrics.get("backgroundCount", 0)),
        "evaluationMode": "synthetic_holdout_only",
        "trainingBackend": "synthetic_baseline",
        "trainingDevice": "cpu",
        "splitMode": "synthetic_only",
        "splitCounts": {},
    }


def _reshape_for_torch(x: np.ndarray, head: str) -> np.ndarray:
    if head == "uid_digit":
        return x.reshape((-1, 1, 24, 16)).astype(np.float32)
    return np.transpose(x.reshape((-1, 32, 32, 3)), (0, 3, 1, 2)).astype(np.float32)


def _resolve_backend(requested: str, requested_device: str) -> str:
    if requested != "torch":
        raise RuntimeError("OCR training is locked to the torch backend. CPU/sklearn fallback is disabled.")
    if requested_device != "cuda":
        raise RuntimeError("OCR training is locked to CUDA. CPU execution is disabled.")
    if not _torch_available():
        raise ImportError("Torch backend requested, but torch is not installed.")
    if not _torch_cuda_available():
        raise RuntimeError("Torch CUDA backend requested, but CUDA is not available.")
    return "torch"


def _partition_dataset(
    x: np.ndarray,
    y: np.ndarray,
    sample_splits: List[str],
    split_source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, Dict[str, int]]:
    if split_source == "manifest":
        split_counts = Counter(split for split in sample_splits if split)
        if all(split_counts.get(name, 0) > 0 for name in ("train", "val", "test")):
            split_array = np.array(sample_splits)
            train_mask = split_array == "train"
            val_mask = split_array == "val"
            test_mask = split_array == "test"
            return (
                x[train_mask],
                x[val_mask],
                x[test_mask],
                y[train_mask],
                y[val_mask],
                y[test_mask],
                "manifest",
                {name: int(split_counts.get(name, 0)) for name in ("train", "val", "test")},
            )
        raise ValueError("Manifest splits are missing train/val/test samples for the filtered dataset.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=_stratify_target(y),
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=_stratify_target(y_temp),
    )
    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        "random",
        {"train": int(x_train.shape[0]), "val": int(x_val.shape[0]), "test": int(x_test.shape[0])},
    )


def _train_sklearn_classifier(
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    output_model_path: Path,
    output_labels_path: Path,
    sample_splits: List[str],
    split_source: str,
) -> Dict[str, Any]:
    if x.shape[0] < 10:
        raise ValueError("Not enough samples for real training.")
    values, counts = np.unique(y, return_counts=True)
    if values.size < 2:
        raise ValueError("Need at least two classes for real training.")

    x_train, x_val, x_test, y_train, y_val, y_test, split_mode, split_counts = _partition_dataset(
        x=x,
        y=y,
        sample_splits=sample_splits,
        split_source=split_source,
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
    model_label_names = [labels[int(class_index)] for class_index in clf.classes_]
    with output_labels_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"labels": model_label_names, "classIds": [int(class_index) for class_index in clf.classes_]},
            fh,
            ensure_ascii=True,
            indent=2,
        )
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
        "evaluationMode": "real_holdout",
        "trainingBackend": "sklearn_logreg",
        "trainingDevice": "cpu",
        "splitMode": split_mode,
        "splitCounts": split_counts,
    }


def _train_torch_classifier(
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    output_model_path: Path,
    output_labels_path: Path,
    head: str,
    sample_splits: List[str],
    split_source: str,
    requested_device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if x.shape[0] < 10:
        raise ValueError("Not enough samples for real training.")
    values, counts = np.unique(y, return_counts=True)
    if values.size < 2:
        raise ValueError("Need at least two classes for real training.")

    images = _reshape_for_torch(x, head=head)
    x_train, x_val, x_test, y_train, y_val, y_test, split_mode, split_counts = _partition_dataset(
        x=images,
        y=y,
        sample_splits=sample_splits,
        split_source=split_source,
    )

    device_name = requested_device.strip().lower()
    if device_name == "auto":
        device_name = "cuda"
    if device_name != "cuda":
        raise RuntimeError("OCR training is locked to CUDA. CPU execution is disabled.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested, but torch.cuda.is_available() is false.")
    device = torch.device(device_name)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    input_channels = int(images.shape[1])
    num_classes = len(labels)

    class UidDigitClassifier(nn.Module):
        def __init__(self, channels: int, classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(96 * 6 * 4, 192),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(192, classes),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            features = self.features(inputs)
            return self.classifier(features)

    class AgentIconClassifier(nn.Module):
        def __init__(self, channels: int, classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.35),
                nn.Linear(512, classes),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            features = self.features(inputs)
            return self.classifier(features)

    if head == "uid_digit":
        model = UidDigitClassifier(channels=input_channels, classes=num_classes).to(device)
    else:
        model = AgentIconClassifier(channels=input_channels, classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max(1e-5, float(learning_rate)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))

    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / max(float(np.mean(class_weights)), 1e-6)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device=device, dtype=torch.float32))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=max(8, int(batch_size)),
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=max(8, int(batch_size)),
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    def _predict_probabilities(loader: DataLoader) -> np.ndarray:
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for batch_inputs, _ in loader:
                batch_inputs = batch_inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                logits = model(batch_inputs)
                probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                outputs.append(probs)
        return np.vstack(outputs) if outputs else np.empty((0, num_classes), dtype=np.float32)

    best_val_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(max(1, int(epochs))):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            batch_targets = batch_targets.to(device=device, dtype=torch.long, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_probs = _predict_probabilities(val_loader)
        val_preds = np.argmax(val_probs, axis=1) if val_probs.size else np.empty((0,), dtype=np.int64)
        val_accuracy = float(accuracy_score(y_val, val_preds)) if val_preds.size else 0.0
        if val_accuracy > best_val_score:
            best_val_score = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_test.shape[0], max(1, int(batch_size))):
            batch_np = x_test[start : start + max(1, int(batch_size))]
            batch_tensor = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            logits = model(batch_tensor)
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            probabilities.append(batch_probs)
    probs = np.vstack(probabilities) if probabilities else np.empty((0, num_classes), dtype=np.float32)
    preds = np.argmax(probs, axis=1) if probs.size else np.empty((0,), dtype=np.int64)

    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0.0))
    precision = float(precision_score(y_test, preds, average="macro", zero_division=0.0))
    recall = float(recall_score(y_test, preds, average="macro", zero_division=0.0))
    ece = _expected_calibration_error(y_test, probs)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.from_numpy(x_train[:1]).to(device=device, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_model_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    with output_labels_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"labels": labels, "classIds": list(range(len(labels)))},
            fh,
            ensure_ascii=True,
            indent=2,
        )
        fh.write("\n")

    latency_p50, latency_p95 = _onnx_latency_stats(output_model_path, x_val)
    return {
        "accuracy": accuracy,
        "macroF1": macro_f1,
        "precision": precision,
        "recall": recall,
        "ece": ece,
        "latencyMsP50": latency_p50,
        "latencyMsP95": latency_p95,
        "backgroundCount": 0,
        "evaluationMode": "real_holdout",
        "trainingBackend": "torch_cnn",
        "trainingDevice": device.type,
        "splitMode": split_mode,
        "splitCounts": split_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train OCR UID and agent-icon heads from manifest data on CUDA only.")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--metrics-file", default="docs/model_metrics.json")
    parser.add_argument("--min-real-samples", type=int, default=2000)
    parser.add_argument("--label-source", choices=["reviewed", "suggested", "any"], default="reviewed")
    parser.add_argument("--split-source", choices=["manifest", "random"], default="manifest")
    parser.add_argument("--backend", choices=["torch"], default="torch")
    parser.add_argument("--torch-device", choices=["cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--data-version", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_file).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    uid_dataset = _load_dataset(
        manifest_path=manifest_path,
        head="uid_digit",
        label_source=args.label_source,
        split_source=args.split_source,
    )
    icon_dataset = _load_dataset(
        manifest_path=manifest_path,
        head="agent_icon",
        label_source=args.label_source,
        split_source=args.split_source,
    )

    _resolve_backend(args.backend, args.torch_device)
    uid_real = uid_dataset.x.shape[0] >= max(20, args.min_real_samples) and len(uid_dataset.label_names) >= 2
    icon_real = icon_dataset.x.shape[0] >= max(20, args.min_real_samples) and len(icon_dataset.label_names) >= 2

    missing_real_heads: List[str] = []
    if not uid_real:
        missing_real_heads.append("uid_digit")
    if not icon_real:
        missing_real_heads.append("agent_icon")
    if missing_real_heads:
        raise RuntimeError(
            "CUDA-only OCR training requires reviewed real datasets for every head. "
            f"Missing trainable heads: {', '.join(missing_real_heads)}."
        )

    uid_metrics = _train_torch_classifier(
        x=uid_dataset.x,
        y=uid_dataset.y,
        labels=uid_dataset.label_names,
        output_model_path=output_dir / "uid_digit.onnx",
        output_labels_path=output_dir / "uid_digit.labels.json",
        head="uid_digit",
        sample_splits=uid_dataset.sample_splits,
        split_source=uid_dataset.split_mode,
        requested_device=args.torch_device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    uid_mode = "real"

    icon_metrics = _train_torch_classifier(
        x=icon_dataset.x,
        y=icon_dataset.y,
        labels=icon_dataset.label_names,
        output_model_path=output_dir / "agent_icon.onnx",
        output_labels_path=output_dir / "agent_icon.labels.json",
        head="agent_icon",
        sample_splits=icon_dataset.sample_splits,
        split_source=icon_dataset.split_mode,
        requested_device=args.torch_device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    icon_mode = "real"
    icon_trained_labels = list(icon_dataset.label_names)

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
        "backends": {
            "uidDigit": str(uid_metrics.get("trainingBackend") or "unknown"),
            "agentIcon": str(icon_metrics.get("trainingBackend") or "unknown"),
        },
    }
    with (output_dir / "model_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(model_manifest, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    metrics_payload = {
        "uid_digit_model": {
            **uid_metrics,
            "dataVersion": data_version,
            "trainedAt": model_manifest["trainedAt"],
            "recordCount": int(uid_dataset.x.shape[0]),
            "skippedRecords": uid_dataset.skipped_records,
            "mode": uid_mode,
            "labelSource": args.label_source,
        },
        "agent_icon_model": {
            **icon_metrics,
            "dataVersion": data_version,
            "trainedAt": model_manifest["trainedAt"],
            "recordCount": int(icon_dataset.x.shape[0]),
            "skippedRecords": icon_dataset.skipped_records,
            "mode": icon_mode,
            "labelSource": args.label_source,
            "rosterAgentCount": len(current_agent_ids()),
            "trainedAgentCount": sum(1 for label in icon_trained_labels if label in set(current_agent_ids())),
            "missingRosterAgents": [agent for agent in current_agent_ids() if agent not in set(icon_trained_labels)],
        },
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(json.dumps({"metrics": metrics_payload, "modelManifest": model_manifest}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
