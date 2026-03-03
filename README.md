# Inter-Knot-Arena-OCR_Scan

OCR and parsing package for VerifierApp full roster scan.

## Scope

- Detect UID from profile/account screens.
- Detect owned agents and progression values.
- Parse amplifiers and discs from inventory/equipment UI.
- Emit normalized payload consumed by VerifierApp.

## Output contract

See `contracts/ocr-output.schema.json`.

## Non-goals

- No direct API access to game backend.
- No game process memory interaction.

## Status

Scaffold initialized.
