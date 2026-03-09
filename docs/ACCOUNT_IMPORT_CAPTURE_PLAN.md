# Account Import Capture Plan

## Purpose

Build OCR data only for Verifier account import:

- `uid_panel` -> `uid_digit`
- `roster` -> owned agent icons
- `agent_detail` -> agent identity and progression context
- `equipment` -> amplifiers, discs, levels, stats
- `disk_detail` -> one screenshot per slot when detailed disc stats are visible
- `amplifier_detail` -> dedicated amplifier detail screen
- `mindscape` -> derive from `agent_detail` footer when visible; dedicated `mindscape` screen is optional provenance only

## Do not use

- generic battle gameplay
- draft footage
- in-run HUD clips
- banned-agent / match-monitor evidence

Those belong to `Inter-Knot Arena CV`, not OCR roster import.

## Required capture lanes

1. `uid_panel`
2. `roster`
3. `agent_detail`
4. `equipment`

## Minimal live-capture commands

```powershell
python scripts/session_capture.py --manifest dataset_manifest.json --head uid_digit --workflow account_import --screen-role uid_panel --duration-sec 60 --fps 1.0 --locale RU --resolution 1080p
python scripts/session_capture.py --manifest dataset_manifest.json --head agent_icon --workflow account_import --screen-role roster --duration-sec 120 --fps 1.0 --locale RU --resolution 1080p
python scripts/session_capture.py --manifest dataset_manifest.json --head agent_icon --workflow account_import --screen-role agent_detail --duration-sec 120 --fps 1.0 --locale RU --resolution 1080p
python scripts/session_capture.py --manifest dataset_manifest.json --head equipment --workflow account_import --screen-role equipment --duration-sec 180 --fps 0.5 --locale RU --resolution 1080p
```

## Screenshot drop-folder path

If the account screens already exist as PNG/JPG files, use the screenshot importer instead of recapturing them:

```powershell
python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\IKA_DATA\ocr\drops\batch_001 --locale RU
```

Supported buckets:

- `uid`
- `roster`
- `agent_detail`
- `equipment`
- `disk1` ... `disk6`
- `amplificator` or `amplifier`
- `mindscape` (optional provenance only)

If screenshots are saved as a generic ordered series, materialize them first:

```powershell
python scripts/materialize_ordered_account_batch.py --input-root C:\Users\loval\OneDrive\Pictures\Screenshots\dataset --output-root D:\IKA_DATA\ocr\drops\batch_001 --start-index 129 --end-index 255 --agent "Е Шуньгуан" --agent "Джейн Доу"
python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\IKA_DATA\ocr\drops\batch_001 --locale RU
```

## Empty slot handling

Treat `equipment` as the authoritative equipped-state screen.

- If a slot is empty on `equipment`, do not capture `diskN` for that slot.
- If no amplifier is equipped, do not capture `amplificator`.
- Ignore recommendation screens that appear after clicking an empty slot. They do not describe the equipped build and must stay out of the OCR dataset.

Practical examples:

- Billy Kid with empty `2,3,5` -> capture `disk1`, `disk4`, `disk6` only
- Lucy de Montefio with empty `1,2,4` -> capture `disk3`, `disk5`, `disk6` only
- Nicole Demara with all disks empty -> no `disk1..disk6`
- Grace Howard with empty `disk2` and no amplifier -> capture `disk1`, `disk3`, `disk4`, `disk5`, `disk6`, no `amplificator`
- Ben Bigger with all disks empty -> no `disk1..disk6`

## Acceptance baseline

- OCR manifest review queues must be built from `workflow=account_import` only.
- `uid_digit`, `agent_icon`, and `equipment` all need dedicated account-screen captures.
- Mindscape truth should come from the left-lower `agent_detail` footer counter, not the nearby unrelated footer counter.
- No combat-only source should affect OCR QA, sampling, or training metrics.
