# Inter-Knot-Arena-OCR_Scan

OCR and parsing package for VerifierApp full roster scan.

## Scope

- Detect UID from profile/account screens.
- Detect owned agents and progression values.
- Parse amplifiers and discs from inventory/equipment UI.
- Emit normalized payload consumed by VerifierApp.

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

## Quick run

```powershell
pip install -r requirements.txt
python scripts/run_scan.py --seed demo --region EU --full-sync
```

Strict mode with explicit lock + anchors:

```powershell
python scripts/run_scan.py --input-lock --anchor-profile --anchor-agents --anchor-equipment --locale RU --resolution 1440p
```

## Dataset workflow (private raw data)

Raw frames/crops are intentionally excluded from git. Use scripts to keep only manifest/config in repo:

```powershell
python scripts/bootstrap_dataset.py --storage-root D:\IKA_DATA\ocr
python scripts/split_dataset.py --manifest dataset_manifest.json --seed 42
```

## Non-goals

- No direct API access to game backend.
- No game process memory interaction.
