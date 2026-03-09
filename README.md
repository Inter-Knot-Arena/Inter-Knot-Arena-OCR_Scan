# Inter-Knot-Arena-OCR_Scan

OCR and parsing package for VerifierApp full roster scan.

## Scope

- Detect UID from profile/account screens.
- Detect owned agents and progression values.
- Parse amplifiers and discs from inventory/equipment UI.
- Derive mindscape from the lower-left agent detail footer when that footer is visible.
- Emit normalized payload consumed by VerifierApp.
- OCR dataset scope is `account import` only. Match gameplay footage is not treated as roster-import ground truth.

## Output contract

See `contracts/ocr-output.schema.json`.

## Runtime pipeline (v1)

- `scanner/pipeline.py` exposes:
  - `scan_roster(session_context, calibration, locale, resolution) -> OcrOutput`
  - `ScanFailure` with fail codes:
    - `ANCHOR_MISMATCH`
    - `UNSUPPORTED_LAYOUT`
    - `LOW_CONFIDENCE`
    - `INPUT_LOCK_NOT_ACTIVE`
- `scripts/run_scan.py` can run either deterministic demo mode (`run_scan`) or strict contract mode (`scan_roster`).
- `scanner/model_runtime.py` loads ONNX models (`uid_digit`, `agent_icon`) with DirectML -> CPU fallback.
- No synthetic UID fallback in strict mode: if UID cannot be extracted with confidence, output is `LOW_CONFIDENCE`.

## Quick run

```powershell
pip install -r requirements.txt
python scripts/run_scan.py --seed demo --region EU --full-sync
```

Strict mode with explicit lock + anchors:

```powershell
python scripts/run_scan.py --input-lock --anchor-profile --anchor-agents --anchor-equipment --locale RU --resolution 1440p
```

Train synthetic baseline models:

```powershell
pip install -r requirements.txt
python scripts/train_synthetic_models.py --output-dir models --metrics-file docs/model_metrics.json
```

Train production OCR heads from manifest data (with fallback to synthetic baseline):

```powershell
python scripts/train_ocr_models.py --manifest dataset_manifest.json --output-dir models --metrics-file docs/model_metrics.json --model-version ocr-heads-v1.3
```

Domain-adaptive training with private live backgrounds:

```powershell
python scripts/train_synthetic_models.py --output-dir models --metrics-file docs/model_metrics.json --uid-background-dir D:\IKA_DATA\ika_live_backgrounds\ocr_uid --icon-background-dir D:\IKA_DATA\ika_live_backgrounds\ocr_icon --uid-samples-per-class 1800 --icon-samples-per-class 1800 --uid-background-probability 0.3 --icon-background-probability 0.8
```

Benchmark runtime:

```powershell
python scripts/benchmark_runtime.py --iterations 100
```

## Dataset workflow (private raw data)

Raw frames/crops are intentionally excluded from git. Use scripts to keep only manifest/config in repo:

```powershell
python scripts/bootstrap_dataset.py --storage-root D:\IKA_DATA\ocr
python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\IKA_DATA\ocr\drops\batch_001 --locale RU
python scripts/ingest_public_sources.py --manifest dataset_manifest.json --sources-file D:\IKA_DATA\ocr_sources.json
python scripts/extract_frames.py --manifest dataset_manifest.json --head uid_digit --workflow account_import --screen-role uid_panel --fps 1.0 --scene-aware
python scripts/extract_frames.py --manifest dataset_manifest.json --head agent_icon --workflow account_import --screen-role roster --fps 1.0
python scripts/extract_frames.py --manifest dataset_manifest.json --head equipment --workflow account_import --screen-role equipment --fps 0.5
python scripts/deduplicate_frames.py --manifest dataset_manifest.json --input-dir D:\IKA_DATA\ocr\frames
python scripts/session_capture.py --manifest dataset_manifest.json --head uid_digit --workflow account_import --screen-role uid_panel --duration-sec 180 --fps 1.0 --locale RU --resolution 1080p
python scripts/prune_manifest.py --manifest dataset_manifest.json --drop-source live_session_1772675708 --drop-source live_session_1772675729
python scripts/prelabel_dataset.py --manifest dataset_manifest.json --workflow account_import --confidence-threshold 0.6
python scripts/qa_audit.py --manifest dataset_manifest.json --workflow account_import --output-file docs/qa_report.json --double-review-file docs/double_review_samples.json
python scripts/export_review_pack.py --manifest dataset_manifest.json --status needs_review --output-csv docs/review_queue.csv
# after manual edit of docs/review_queue.csv:
python scripts/apply_review_labels.py --manifest dataset_manifest.json --input-csv docs/review_queue.csv --review-round final --reviewer-id qa_operator_1
python scripts/build_sampling_plan.py --manifest dataset_manifest.json --workflow account_import --target-uid-digit 20000 --target-agent-icon 15000 --target-equipment 15000 --output-file docs/sampling_plan.json
python scripts/build_account_capture_backlog.py --manifest dataset_manifest.json --output-file docs/account_capture_backlog.json
python scripts/split_dataset.py --manifest dataset_manifest.json --seed 42
```

Drop-folder screenshot import is the preferred route for manual account captures. The importer understands these screen buckets from file names or parent folders:

- `uid`
- `roster`
- `agent_detail`
- `equipment`
- `disk1` ... `disk6`
- `amplificator` or `amplifier`
- `mindscape` (optional provenance only; primary mindscape truth should come from `agent_detail`)

Imported files are copied into private storage under `D:\IKA_DATA\ocr\raw\manual_screens\...` and appended to `dataset_manifest.json`.

## Dataset policy

- `workflow=account_import` is the only default scope for OCR training/review artifacts.
- `workflow=combat_reference` may exist in the manifest for provenance, but those records are excluded from OCR review, QA, and sampling by default.
- Use `scripts/realign_import_workflow.py` if an older manifest mixed account-import and gameplay sources.
- `mindscape` should be derived from `agent_detail` when the footer counter is visible. Separate `mindscape` screenshots are optional provenance and stay out of the current import-eligible training slice.

## Non-goals

- No direct API access to game backend.
- No game process memory interaction.
