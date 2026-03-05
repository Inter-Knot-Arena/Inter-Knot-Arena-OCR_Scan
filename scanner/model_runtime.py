from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_MANIFEST_PATH = MODEL_DIR / "model_manifest.json"


def _provider_priority() -> list[str]:
    available = set(ort.get_available_providers())
    preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in preferred if provider in available] or ["CPUExecutionProvider"]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def get_model_metadata(default_version: str) -> Dict[str, str]:
    if not MODEL_MANIFEST_PATH.exists():
        return {"modelVersion": default_version, "dataVersion": "unknown"}
    try:
        payload = _read_json(MODEL_MANIFEST_PATH)
        model_version = str(payload.get("version") or default_version)
        data_version = str(payload.get("dataVersion") or "unknown")
        return {"modelVersion": model_version, "dataVersion": data_version}
    except Exception:
        return {"modelVersion": default_version, "dataVersion": "unknown"}


def _normalize_probability_output(value: object, labels: Sequence[str]) -> Dict[str, float]:
    if isinstance(value, list):
        # Many sklearn ONNX exports return List[Dict[label, score]]
        if value and isinstance(value[0], dict):
            first = value[0]
            probabilities: Dict[str, float] = {}
            for key, score in first.items():
                label = _map_label_key(key, labels)
                probabilities[label] = float(score)
            return probabilities

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            probs = value
        elif value.ndim >= 2:
            probs = value[0]
        else:
            probs = np.array([], dtype=np.float32)
        probs = np.asarray(probs).reshape(-1)
        if probs.size == 1 and len(labels) > 1:
            first_value = float(probs[0])
            if np.issubdtype(probs.dtype, np.integer) or first_value < 0.0 or first_value > 1.0:
                return {}
        output: Dict[str, float] = {}
        for idx, label in enumerate(labels):
            if idx < probs.shape[0]:
                output[label] = float(probs[idx])
        return output

    return {}


def _probability_quality(probabilities: Dict[str, float]) -> tuple[int, int, float, float]:
    if not probabilities:
        return (0, 0, 0.0, 0.0)
    values = [float(value) for value in probabilities.values() if math.isfinite(float(value))]
    if not values:
        return (0, 0, 0.0, 0.0)
    in_unit_range = sum(1 for value in values if 0.0 <= value <= 1.0)
    value_sum = float(sum(values))
    peak = float(max(values))
    return (len(values), in_unit_range, value_sum, peak)


def _map_label_key(raw: object, labels: Sequence[str]) -> str:
    if isinstance(raw, (np.integer, int)):
        idx = int(raw)
        if 0 <= idx < len(labels):
            return labels[idx]
        return str(idx)

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    text = str(raw).strip()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(labels):
            return labels[idx]
    return text


@dataclass(slots=True)
class Prediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]


class OnnxClassifier:
    def __init__(self, model_path: Path, labels_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file missing: {labels_path}")

        labels_payload = _read_json(labels_path)
        labels = labels_payload.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"labels must be non-empty array in {labels_path}")
        self.labels = [str(item) for item in labels]

        providers = _provider_priority()
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, feature_vector: np.ndarray) -> Prediction:
        vector = np.asarray(feature_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        elif vector.ndim != 2:
            raise ValueError("feature_vector must be 1D or 2D")

        raw_outputs = self.session.run(self.output_names, {self.input_name: vector})
        label: str | None = None
        probability_candidates: list[Dict[str, float]] = []

        for output in raw_outputs:
            if label is None:
                if isinstance(output, np.ndarray) and output.size > 0:
                    raw_label = output[0]
                    label = _map_label_key(raw_label, self.labels)
                elif isinstance(output, list) and output:
                    label = _map_label_key(output[0], self.labels)

            candidate = _normalize_probability_output(output, self.labels)
            if candidate:
                probability_candidates.append(candidate)

        probabilities: Dict[str, float] = {}
        if probability_candidates:
            probabilities = max(probability_candidates, key=_probability_quality)

        if label is None:
            # fallback from max-probability
            if probabilities:
                label = max(probabilities.items(), key=lambda item: item[1])[0]
            else:
                label = self.labels[0]

        confidence = float(probabilities.get(label, 0.0))
        if confidence <= 0.0 and probabilities:
            confidence = float(max(probabilities.values()))
        if confidence <= 0.0 and not probabilities:
            confidence = 0.51
        return Prediction(label=label, confidence=confidence, probabilities=probabilities)


class ModelRegistry:
    _lock = threading.Lock()
    _uid_classifier: OnnxClassifier | None = None
    _agent_classifier: OnnxClassifier | None = None

    @classmethod
    def has_uid_model(cls) -> bool:
        return (MODEL_DIR / "uid_digit.onnx").exists() and (MODEL_DIR / "uid_digit.labels.json").exists()

    @classmethod
    def has_agent_model(cls) -> bool:
        return (MODEL_DIR / "agent_icon.onnx").exists() and (MODEL_DIR / "agent_icon.labels.json").exists()

    @classmethod
    def uid_classifier(cls) -> OnnxClassifier:
        with cls._lock:
            if cls._uid_classifier is None:
                cls._uid_classifier = OnnxClassifier(
                    model_path=MODEL_DIR / "uid_digit.onnx",
                    labels_path=MODEL_DIR / "uid_digit.labels.json",
                )
            return cls._uid_classifier

    @classmethod
    def agent_classifier(cls) -> OnnxClassifier:
        with cls._lock:
            if cls._agent_classifier is None:
                cls._agent_classifier = OnnxClassifier(
                    model_path=MODEL_DIR / "agent_icon.onnx",
                    labels_path=MODEL_DIR / "agent_icon.labels.json",
                )
            return cls._agent_classifier


def preprocess_digit(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 24), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(-1)


def preprocess_icon(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(-1)


def classify_uid_digits(digit_images: Iterable[np.ndarray]) -> Tuple[str, float]:
    classifier = ModelRegistry.uid_classifier()
    labels: list[str] = []
    confidences: list[float] = []
    for image in digit_images:
        prediction = classifier.predict(preprocess_digit(image))
        labels.append(prediction.label)
        confidences.append(prediction.confidence)

    uid = "".join(labels)
    confidence = float(sum(confidences) / max(len(confidences), 1))
    return uid, confidence


def classify_agent_icon(image: np.ndarray) -> Prediction:
    classifier = ModelRegistry.agent_classifier()
    return classifier.predict(preprocess_icon(image))
