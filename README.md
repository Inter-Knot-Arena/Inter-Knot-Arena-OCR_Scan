# Inter-Knot-Arena-OCR_Scan

OCR and parsing package for VerifierApp full roster scan.

## Scope

- Detect UID from profile/account screens.
- Detect owned agents and progression values.
- Parse amplifiers and discs from inventory/equipment UI.
- Emit normalized payload consumed by VerifierApp.

## Output contract

See `contracts/ocr-output.schema.json`.

## Runtime stub (v1)

- `scanner/pipeline.py` implements hybrid deterministic placeholder pipeline.
- `scripts/run_scan.py` prints a contract-compliant OCR payload.

## Quick run

```powershell
pip install -r requirements.txt
python scripts/run_scan.py --seed demo --region EU --full-sync
```

## Non-goals

- No direct API access to game backend.
- No game process memory interaction.
