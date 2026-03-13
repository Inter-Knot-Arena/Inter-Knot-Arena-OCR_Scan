# OCR Training Readiness

- Generated at: `2026-03-13T21:33:07Z`
- Workflow: `account_import`
- Scoped records: `6820`
- Reviewed truth records: `3542`

## Current Train Heads

### uid_digit (READY)

- Samples: `2220`
- Distinct labels: `10`
- Splits: `train=1750` `val=190` `test=280` `unsplit=0`
- Missing digits: `none`
- Materialized digit samples: `2220`

### agent_icon (BLOCKED)

- Samples: `987`
- Distinct labels: `49`
- Splits: `train=790` `val=103` `test=94` `unsplit=0`
- Reviewed current agents: `49/49`
- Blockers: `need_at_least_2000_reviewed_samples`

## OCR Field Targets

### uid_panel_full_uid

- Reviewed records: `224`
- Complete records: `224`
- Splits: `train=177` `val=19` `test=28` `unsplit=0`

### agent_detail_level

- Reviewed records: `506`
- Complete records: `1`
- Splits: `train=1` `val=0` `test=0` `unsplit=0`

### agent_detail_mindscape

- Reviewed records: `506`
- Complete records: `1`
- Splits: `train=1` `val=0` `test=0` `unsplit=0`

### agent_detail_stats

- Reviewed records: `506`
- Complete records: `1`
- Splits: `train=1` `val=0` `test=0` `unsplit=0`

### amplifier_detail

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=0` `val=1` `test=0` `unsplit=0`

### disk_detail

- Reviewed records: `6`
- Complete records: `6`
- Splits: `train=4` `val=2` `test=0` `unsplit=0`

### equipment_overview

- Reviewed records: `481`
- Complete records: `480`
- Splits: `train=379` `val=51` `test=50` `unsplit=0`

### roster_owned_agents

- Reviewed records: `104`
- Complete records: `104`
- Splits: `train=86` `val=7` `test=11` `unsplit=0`
