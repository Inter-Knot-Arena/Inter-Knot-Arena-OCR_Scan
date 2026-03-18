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
- `scanner/model_runtime.py` loads ONNX models (`uid_digit`, `agent_icon`, `disk_detail`) with CUDA-only runtime validation.
- `amplifier_detail` is resolved in runtime via title-crop OCR plus the reviewed alias contract in `contracts/amplifier-title-aliases.json`.
- `agent_detail` runtime uses `uid_digit` for level/mindscape digits and `contracts/agent-detail-digit-templates.json` for full stat-row parsing.
- `equipment` overview runtime now contributes pixel `weaponPresent` and `discSlotOccupancy` when the overview capture is confident enough, even before individual amplifier/disk detail screens are parsed.
- No synthetic UID fallback in strict mode: if UID cannot be extracted with confidence, output is `LOW_CONFIDENCE`.
- Strict mode does not inject demo agents or `uidCandidates`: only explicit runtime captures/crops are trusted.
- Multi-page roster captures are normalized into page-aware `agentSlotIndex` values, so downstream detail/equipment captures can bind to a stable roster identity.
- Anchor flags are now treated as explicit contract input and are not auto-promoted from incidental captures.

## Quick run

```powershell
pip install -r requirements.txt
python scripts/run_scan.py --seed demo --region EU --full-sync
```

Strict mode with explicit lock + anchors:

```powershell
python scripts/run_scan.py --input-lock --anchor-profile --anchor-agents --anchor-equipment --locale RU --resolution 1440p
```

Strict mode with richer runtime captures:

```powershell
python scripts/run_scan.py --input-lock --anchor-profile --anchor-agents --anchor-equipment --locale RU --resolution 1080p --screen-capture "agent_detail|D:\shots\agent_detail.png|||1|detail_slot_1" --screen-capture "amplifier_detail|D:\shots\weapon.png|||1|weapon_slot1" --screen-capture "disk_detail|D:\shots\disk_1.png||1|1|disk1_slot1"
```

Run the OCR regression tests:

```powershell
python -m unittest discover -s tests -v
```

Train synthetic baseline models:

```powershell
pip install -r requirements.txt
python scripts/train_synthetic_models.py --output-dir models --metrics-file docs/model_metrics.json
```

Train production OCR heads from manifest data (CUDA-only, reviewed real data only):

```powershell
python scripts/train_ocr_models.py --manifest dataset_manifest.json --output-dir models --metrics-file docs/model_metrics.json --model-version ocr-heads-v1.4 --label-source reviewed --split-source manifest --backend torch --torch-device cuda
```

Build reviewed-truth training readiness report before running real training:

```powershell
python scripts/build_training_readiness.py --manifest dataset_manifest.json --output-json docs/training_readiness.json --output-md docs/training_readiness.md
```

Domain-adaptive training with private live backgrounds:

```powershell
python scripts/train_synthetic_models.py --output-dir models --metrics-file docs/model_metrics.json --uid-background-dir D:\IKA_DATA\ika_live_backgrounds\ocr_uid --icon-background-dir D:\IKA_DATA\ika_live_backgrounds\ocr_icon --uid-samples-per-class 1800 --icon-samples-per-class 1800 --uid-background-probability 0.3 --icon-background-probability 0.8
```

Benchmark runtime:

```powershell
python scripts/benchmark_runtime.py --demo --iterations 100
python scripts/benchmark_runtime.py --iterations 100 --session-context D:\shots\strict_session_context.json --locale RU --resolution 1080p
```

## Dataset workflow (private raw data)

Raw frames/crops are intentionally excluded from git. Use scripts to keep only manifest/config in repo:

```powershell
python scripts/bootstrap_dataset.py --storage-root D:\IKA_DATA\ocr
python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\IKA_DATA\ocr\drops\batch_001 --locale RU
python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\IKA_DATA\ocr\drops\batch_en_001 --locale EN
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
python scripts/materialize_uid_digit_records.py --manifest dataset_manifest.json
python scripts/split_dataset.py --manifest dataset_manifest.json --workflow account_import --import-eligible-only --reviewed-only --seed 42
python scripts/build_sampling_plan.py --manifest dataset_manifest.json --workflow account_import --target-uid-digit 20000 --target-agent-icon 15000 --target-equipment 15000 --output-file docs/sampling_plan.json
python scripts/build_account_capture_backlog.py --manifest dataset_manifest.json --output-file docs/account_capture_backlog.json
python scripts/sync_equipment_taxonomy.py
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

For mixed-language drops, use path hints (`en\...`, `ru\...`, `english\...`, `russian\...`) and import with:

```powershell
python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\IKA_DATA\ocr\drops\batch_mixed_001 --locale MIXED --fallback-locale UNKNOWN
```

If a file has no locale hint in `MIXED` mode, the importer now stores `locale=UNKNOWN` instead of lying about the batch locale.

## Dataset policy

- `workflow=account_import` is the only default scope for OCR training/review artifacts.
- `workflow=combat_reference` may exist in the manifest for provenance, but those records are excluded from OCR review, QA, and sampling by default.
- Use `scripts/realign_import_workflow.py` if an older manifest mixed account-import and gameplay sources.
- `mindscape` should be derived from `agent_detail` when the footer counter is visible. Separate `mindscape` screenshots are optional provenance and stay out of the current import-eligible training slice.
- Reviewed full UID panels are not trainable digit samples by themselves. Materialize them into derived `uid_digit` crops with `scripts/materialize_uid_digit_records.py` before building readiness or training.
- For OCR training, assign splits on the reviewed `account_import` slice instead of the full mixed manifest: `python scripts/split_dataset.py --manifest dataset_manifest.json --workflow account_import --import-eligible-only --reviewed-only --seed 42`.
- Current strict runtime still targets the supported `RU/EN` layout family, but resolution is now inferred from the actual capture and canonicalized to the nearest layout family (`1080p`/`1440p`). Fractional crops removed hard pixel constants, though the pipeline is still not fully UI/layout-agnostic.

## Non-goals

- No direct API access to game backend.
- No game process memory interaction.
