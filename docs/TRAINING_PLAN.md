# OCR Training Plan (Draft)

## Data sources

- Captured UI crops from verifier calibration sessions.
- Synthetic augmentations for scale/blur/noise.

## Model tracks

1. UID numeric OCR model.
2. Agent icon classifier.
3. Equipment parser (discs and amplifiers).

## Quality gates

- UID exact match >= 99.5%
- Agent id top-1 >= 99.0%
- Disc/amplifier parse F1 >= 97.0%

## Runtime constraints

- Single-pass inference under 500 ms per screen on GTX 970.
