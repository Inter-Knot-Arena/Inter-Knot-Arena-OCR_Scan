# OCR Training Plan (v1)

## Private dataset layout

- `raw/`: videos and screenshots (private local storage only)
- `frames/`: extracted frames for training
- `labels_uid/`: UID annotations
- `labels_agents/`: agent/icon annotations
- `labels_equipment/`: amplifier and disc annotations

Only `dataset_manifest.json`, scripts, configs, metrics, and release artifacts are stored in git.

## Data provenance

`dataset_manifest.json` records:

- source id
- source URL or origin label
- capture date
- license note

## Model tracks

1. UID OCR head (lightweight ONNX baseline, synthetic-trained logistic classifier in v1 repo state)
2. Agent icon classifier (synthetic-trained ONNX baseline in v1 repo state)
3. Equipment heads:
   - amplifier id
   - disc set id
   - disc slot and level

## Confidence policy

- Per-field confidence map is required.
- Temperature scaling is applied per output head.
- Low-confidence outputs must include `lowConfReasons[]`.

## Quality gates

- UID exact match >= 99.5% on holdout RU+EN.
- Agent top-1 >= 99.0%.
- Equipment parse F1 >= 97.0%.

## Runtime targets

- Single pass per screen <= 500 ms on GTX 970 class hardware.
- CPU fallback must remain functional.

## Training scripts in repo

- `scripts/train_synthetic_models.py`: trains + exports ONNX (`models/uid_digit.onnx`, `models/agent_icon.onnx`).
- `scripts/train_synthetic_models.py` supports private live-frame backgrounds:
  - `--uid-background-dir`
  - `--icon-background-dir`
  - plus sample-size and background-probability controls per head for faster iteration.
- `scripts/benchmark_runtime.py`: latency benchmark of `run_scan`.
