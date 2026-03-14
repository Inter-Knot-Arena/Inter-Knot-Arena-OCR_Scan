# OCR Training Readiness

- Generated at: `2026-03-14T12:07:45Z`
- Workflow: `account_import`
- Scoped records: `6820`
- Reviewed truth records: `6483`

## Current Train Heads

### uid_digit (READY)

- Samples: `2220`
- Distinct labels: `10`
- Splits: `train=1850` `val=170` `test=200` `unsplit=0`
- Missing digits: `none`
- Materialized digit samples: `2220`

### agent_icon (BLOCKED)

- Samples: `987`
- Distinct labels: `49`
- Splits: `train=802` `val=94` `test=91` `unsplit=0`
- Reviewed current agents: `49/49`
- Blockers: `need_at_least_2000_reviewed_samples`

## OCR Field Targets

### uid_panel_full_uid

- Reviewed records: `224`
- Complete records: `224`
- Splits: `train=187` `val=17` `test=20` `unsplit=0`

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

- Reviewed records: `451`
- Complete records: `451`
- Splits: `train=355` `val=51` `test=45` `unsplit=0`

### disk_detail

- Reviewed records: `2497`
- Complete records: `2497`
- Splits: `train=1985` `val=250` `test=262` `unsplit=0`

### equipment_overview

- Reviewed records: `481`
- Complete records: `480`
- Splits: `train=390` `val=41` `test=49` `unsplit=0`

### roster_owned_agents

- Reviewed records: `104`
- Complete records: `104`
- Splits: `train=81` `val=14` `test=9` `unsplit=0`
