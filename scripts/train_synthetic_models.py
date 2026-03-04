from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

RNG = np.random.default_rng(42)
AGENT_LABELS = [
    "agent_anby",
    "agent_nicole",
    "agent_ellen",
    "agent_lycaon",
    "agent_koleda",
    "agent_vivian",
]


def _render_digit_sample(digit: int) -> np.ndarray:
    canvas = np.zeros((24, 16), dtype=np.uint8)
    font = int(RNG.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]))
    scale = float(RNG.uniform(0.55, 0.9))
    thickness = int(RNG.integers(1, 3))
    text = str(digit)
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(max(0, min(15 - text_w, RNG.integers(-1, 4))))
    y = int(max(text_h + 1, min(23, RNG.integers(text_h + 1, 23))))
    cv2.putText(canvas, text, (x, y), font, scale, int(RNG.integers(180, 255)), thickness, cv2.LINE_AA)
    noise = RNG.normal(0, 11, size=canvas.shape).astype(np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _agent_color(label: str) -> Tuple[int, int, int]:
    digest = abs(hash(label))
    return (
        80 + digest % 120,
        80 + (digest // 7) % 120,
        80 + (digest // 13) % 120,
    )


def _render_agent_icon(label: str) -> np.ndarray:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    color = _agent_color(label)
    img[:] = (color[0] // 3, color[1] // 3, color[2] // 3)

    idx = AGENT_LABELS.index(label)
    center = (16, 16)
    radius = 5 + (idx % 6)
    cv2.circle(img, center, radius, color, thickness=-1)
    cv2.line(img, (4, 4 + idx), (28, 28 - idx), (255 - color[0], 255 - color[1], 255 - color[2]), 2)
    cv2.rectangle(img, (2 + idx % 6, 24 - idx % 6), (10 + idx % 6, 30), color[::-1], thickness=-1)

    noise = RNG.normal(0, 7, size=img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _fit_and_export(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    model_path: Path,
    labels_path: Path,
) -> Dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = LogisticRegression(
        max_iter=800,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    initial_type = [("input", FloatTensorType([None, features.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=17)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        fh.write(onnx_model.SerializeToString())

    labels_payload = {"labels": label_names}
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump(labels_payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    return {"accuracy": accuracy}


def train_uid_model(output_dir: Path) -> Dict[str, float]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for digit in range(10):
        for _ in range(900):
            sample = _render_digit_sample(digit)
            features.append(sample.astype(np.float32).reshape(-1) / 255.0)
            labels.append(digit)
    x = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return _fit_and_export(
        features=x,
        labels=y,
        label_names=[str(i) for i in range(10)],
        model_path=output_dir / "uid_digit.onnx",
        labels_path=output_dir / "uid_digit.labels.json",
    )


def train_agent_icon_model(output_dir: Path) -> Dict[str, float]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for index, label in enumerate(AGENT_LABELS):
        for _ in range(1000):
            sample = _render_agent_icon(label)
            features.append(sample.astype(np.float32).reshape(-1) / 255.0)
            labels.append(index)
    x = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return _fit_and_export(
        features=x,
        labels=y,
        label_names=AGENT_LABELS,
        model_path=output_dir / "agent_icon.onnx",
        labels_path=output_dir / "agent_icon.labels.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train synthetic ONNX OCR models for UID and agent icons.")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--metrics-file", default="docs/model_metrics.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    uid_metrics = train_uid_model(output_dir)
    icon_metrics = train_agent_icon_model(output_dir)

    metrics = {
        "uid_digit_model": uid_metrics,
        "agent_icon_model": icon_metrics,
    }

    metrics_path = Path(args.metrics_file).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=True, indent=2)
        fh.write("\n")

    print(f"Models written to: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
