# Account Import Capture Plan

## Purpose

Build OCR data only for Verifier account import:

- `uid_panel` -> `uid_digit`
- `roster` -> owned agent icons
- `agent_detail` -> agent identity and progression context
- `equipment` -> amplifiers, discs, levels, stats

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

## Acceptance baseline

- OCR manifest review queues must be built from `workflow=account_import` only.
- `uid_digit`, `agent_icon`, and `equipment` all need dedicated account-screen captures.
- No combat-only source should affect OCR QA, sampling, or training metrics.
