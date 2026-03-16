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
_CUDA_DLLS_PRELOADED = False


def _preload_cuda_runtime_dlls() -> None:
    global _CUDA_DLLS_PRELOADED
    if _CUDA_DLLS_PRELOADED:
        return
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload()
    _CUDA_DLLS_PRELOADED = True


def _provider_priority() -> list[str]:
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        available_list = ", ".join(sorted(available)) or "none"
        raise RuntimeError(
            "CUDAExecutionProvider is required for OCR runtime. "
            f"Available providers: {available_list}."
        )
    return ["CUDAExecutionProvider"]


def _create_cuda_session(model_path: Path) -> ort.InferenceSession:
    _preload_cuda_runtime_dlls()
    session = ort.InferenceSession(str(model_path), providers=_provider_priority())
    actual = list(session.get_providers())
    if "CUDAExecutionProvider" not in actual:
        actual_list = ", ".join(actual) or "none"
        raise RuntimeError(
            "OCR runtime requested CUDAExecutionProvider, but ONNX Runtime created a non-CUDA session. "
            f"Actual providers: {actual_list}. Check CUDA/cuDNN/MSVC runtime dependencies."
        )
    return session


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


def _normalize_probability_output(value: object, labels: Sequence[str], class_id_map: Dict[int, str]) -> Dict[str, float]:
    if isinstance(value, list):
        # Many sklearn ONNX exports return List[Dict[label, score]]
        if value and isinstance(value[0], dict):
            first = value[0]
            probabilities: Dict[str, float] = {}
            for key, score in first.items():
                label = _map_label_key(key, labels, class_id_map)
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
        if probs.size > 1:
            total = float(np.sum(probs))
            looks_like_probabilities = bool(
                np.all(np.isfinite(probs))
                and np.all(probs >= 0.0)
                and np.all(probs <= 1.0)
                and 0.95 <= total <= 1.05
            )
            if not looks_like_probabilities:
                shifted = probs - float(np.max(probs))
                exp_values = np.exp(shifted)
                denom = float(np.sum(exp_values))
                if denom > 0.0:
                    probs = exp_values / denom
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


def _map_label_key(raw: object, labels: Sequence[str], class_id_map: Dict[int, str]) -> str:
    if isinstance(raw, (np.integer, int)):
        idx = int(raw)
        mapped = class_id_map.get(idx)
        if mapped:
            return mapped
        if 0 <= idx < len(labels):
            return labels[idx]
        return str(idx)

    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")

    text = str(raw).strip()
    if text.isdigit():
        idx = int(text)
        mapped = class_id_map.get(idx)
        if mapped:
            return mapped
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
        class_ids_raw = labels_payload.get("classIds")
        self.class_id_map: Dict[int, str] = {}
        if isinstance(class_ids_raw, list) and len(class_ids_raw) == len(self.labels):
            for class_id_value, label in zip(class_ids_raw, self.labels):
                if isinstance(class_id_value, (int, np.integer)):
                    self.class_id_map[int(class_id_value)] = label

        self.session = _create_cuda_session(model_path)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = tuple(input_meta.shape)
        self.output_names = [output.name for output in self.session.get_outputs()]

    def expects_image_input(self) -> bool:
        return len(self.input_shape) == 4

    def _expected_channel_count(self) -> int | None:
        if not self.expects_image_input():
            return None
        for raw_value in (self.input_shape[1], self.input_shape[-1]):
            if isinstance(raw_value, (int, np.integer)) and int(raw_value) in {1, 3}:
                return int(raw_value)
        return None

    def _expects_nchw(self) -> bool:
        if not self.expects_image_input():
            return False
        value = self.input_shape[1]
        return isinstance(value, (int, np.integer)) and int(value) in {1, 3}

    def _expects_nhwc(self) -> bool:
        if not self.expects_image_input():
            return False
        value = self.input_shape[-1]
        return isinstance(value, (int, np.integer)) and int(value) in {1, 3}

    def _prepare_input(self, sample: np.ndarray) -> np.ndarray:
        array = np.asarray(sample, dtype=np.float32)
        if not self.expects_image_input():
            if array.ndim == 1:
                return array.reshape(1, -1)
            if array.ndim == 2:
                if array.shape[0] == 1:
                    return array
                return array.reshape(1, -1)
            return array.reshape(1, -1)

        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim == 3:
            channel_count = self._expected_channel_count()
            if self._expects_nchw():
                if channel_count is not None and array.shape[0] != channel_count and array.shape[-1] == channel_count:
                    array = np.transpose(array, (2, 0, 1))
                return array.reshape(1, *array.shape)
            if self._expects_nhwc():
                if channel_count is not None and array.shape[-1] != channel_count and array.shape[0] == channel_count:
                    array = np.transpose(array, (1, 2, 0))
                return array.reshape(1, *array.shape)
            return array.reshape(1, *array.shape)
        if array.ndim == 4:
            channel_count = self._expected_channel_count()
            if self._expects_nchw() and channel_count is not None and array.shape[1] != channel_count and array.shape[-1] == channel_count:
                array = np.transpose(array, (0, 3, 1, 2))
            elif self._expects_nhwc() and channel_count is not None and array.shape[-1] != channel_count and array.shape[1] == channel_count:
                array = np.transpose(array, (0, 2, 3, 1))
            return array
        raise ValueError("sample must be 1D, 2D, 3D, or 4D")

    def predict(self, sample: np.ndarray) -> Prediction:
        prepared = self._prepare_input(sample)
        raw_outputs = self.session.run(self.output_names, {self.input_name: prepared})
        label: str | None = None
        probability_candidates: list[Dict[str, float]] = []

        for output in raw_outputs:
            if label is None:
                if isinstance(output, np.ndarray) and np.asarray(output).size == 1:
                    raw_label = np.asarray(output).reshape(-1)[0]
                    label = _map_label_key(raw_label, self.labels, self.class_id_map)
                elif isinstance(output, list) and output and not isinstance(output[0], dict):
                    label = _map_label_key(output[0], self.labels, self.class_id_map)

            candidate = _normalize_probability_output(output, self.labels, self.class_id_map)
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
    _disk_classifier: OnnxClassifier | None = None

    @classmethod
    def has_uid_model(cls) -> bool:
        return (MODEL_DIR / "uid_digit.onnx").exists() and (MODEL_DIR / "uid_digit.labels.json").exists()

    @classmethod
    def has_agent_model(cls) -> bool:
        return (MODEL_DIR / "agent_icon.onnx").exists() and (MODEL_DIR / "agent_icon.labels.json").exists()

    @classmethod
    def has_disk_model(cls) -> bool:
        return (MODEL_DIR / "disk_detail.onnx").exists() and (MODEL_DIR / "disk_detail.labels.json").exists()

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

    @classmethod
    def disk_classifier(cls) -> OnnxClassifier:
        with cls._lock:
            if cls._disk_classifier is None:
                cls._disk_classifier = OnnxClassifier(
                    model_path=MODEL_DIR / "disk_detail.onnx",
                    labels_path=MODEL_DIR / "disk_detail.labels.json",
                )
            return cls._disk_classifier


def preprocess_digit(image: np.ndarray) -> np.ndarray:
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
    normalized = canvas.astype(np.float32) / 255.0
    return normalized


def segment_uid_digits(image: np.ndarray) -> List[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    column_counts = np.count_nonzero(threshold, axis=0)
    min_column_pixels = max(2, int(round(threshold.shape[0] * 0.12)))
    boxes: list[tuple[int, int, int, int]] = []

    start: int | None = None
    for column_index, pixel_count in enumerate(column_counts):
        if pixel_count >= min_column_pixels:
            if start is None:
                start = column_index
            continue
        if start is None:
            continue
        end = column_index - 1
        segment = threshold[:, start : end + 1]
        ys, xs = np.nonzero(segment)
        if xs.size:
            y0 = int(ys.min())
            y1 = int(ys.max())
            boxes.append((start, y0, end - start + 1, y1 - y0 + 1))
        start = None

    if start is not None:
        segment = threshold[:, start:]
        ys, xs = np.nonzero(segment)
        if xs.size:
            y0 = int(ys.min())
            y1 = int(ys.max())
            boxes.append((start, y0, threshold.shape[1] - start, y1 - y0 + 1))

    digits: list[np.ndarray] = []
    height, width = threshold.shape[:2]
    for x, y, contour_width, contour_height in boxes:
        if contour_width < 3:
            continue
        if contour_height < int(round(height * 0.45)):
            continue
        if contour_width > int(round(width * 0.2)):
            continue
        pad = 1
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + contour_width + pad)
        y1 = min(height, y + contour_height + pad)
        crop = threshold[y0:y1, x0:x1]
        if crop.size:
            digits.append(crop)
    if len(digits) == 10:
        return digits

    # Some runtime/materialized crops already isolate the full 10-digit band.
    # In that case equal-width slicing is more reliable than component splitting.
    height, width = threshold.shape[:2]
    if height <= 40 and width >= 120 and (float(width) / float(max(height, 1))) >= 4.5:
        ys, xs = np.nonzero(threshold)
        if xs.size and ys.size:
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            band = threshold[y0:y1, x0:x1]
            if band.size:
                band_height, band_width = band.shape[:2]
                slices: list[np.ndarray] = []
                for digit_index in range(10):
                    left = int(round(band_width * digit_index / 10.0))
                    right = int(round(band_width * (digit_index + 1) / 10.0))
                    right = max(right, left + 1)
                    slice_crop = band[:, left:right]
                    if slice_crop.size == 0:
                        break
                    inner_ys, inner_xs = np.nonzero(slice_crop)
                    if inner_xs.size:
                        sx0 = int(inner_xs.min())
                        sx1 = int(inner_xs.max()) + 1
                        sy0 = int(inner_ys.min())
                        sy1 = int(inner_ys.max()) + 1
                        slice_crop = slice_crop[sy0:sy1, sx0:sx1]
                    if slice_crop.size == 0:
                        break
                    slices.append(slice_crop)
                if len(slices) == 10:
                    return slices
    return digits


def preprocess_digit_for_classifier(image: np.ndarray, classifier: OnnxClassifier) -> np.ndarray:
    normalized = preprocess_digit(image)
    if classifier.expects_image_input():
        return normalized
    return normalized.reshape(-1)


def preprocess_icon(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def preprocess_icon_for_classifier(image: np.ndarray, classifier: OnnxClassifier) -> np.ndarray:
    normalized = preprocess_icon(image)
    if classifier.expects_image_input():
        return normalized
    return normalized.reshape(-1)


def classify_uid_digits(digit_images: Iterable[np.ndarray]) -> Tuple[str, float]:
    classifier = ModelRegistry.uid_classifier()
    labels: list[str] = []
    confidences: list[float] = []
    for image in digit_images:
        prediction = classifier.predict(preprocess_digit_for_classifier(image, classifier))
        labels.append(prediction.label)
        confidences.append(prediction.confidence)

    uid = "".join(labels)
    confidence = float(sum(confidences) / max(len(confidences), 1))
    return uid, confidence


def classify_agent_icon(image: np.ndarray) -> Prediction:
    classifier = ModelRegistry.agent_classifier()
    return classifier.predict(preprocess_icon_for_classifier(image, classifier))


def classify_disk_detail(image: np.ndarray) -> Prediction:
    classifier = ModelRegistry.disk_classifier()
    return classifier.predict(preprocess_icon_for_classifier(image, classifier))
