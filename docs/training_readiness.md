# OCR Training Readiness

- Generated at: `2026-03-13T20:03:59Z`
- Workflow: `account_import`
- Scoped records: `4610`
- Reviewed truth records: `1109`

## Current Train Heads

### uid_digit (BLOCKED)

- Samples: `26`
- Distinct labels: `6`
- Splits: `train=25` `val=1` `test=0` `unsplit=0`
- Missing digits: `2, 4, 8, 9`
- Materialized digit samples: `10`
- Blockers: `need_at_least_2000_reviewed_samples; need_at_least_10_distinct_labels; missing_test_split_samples`

### agent_icon (BLOCKED)

- Samples: `987`
- Distinct labels: `49`
- Splits: `train=798` `val=93` `test=96` `unsplit=0`
- Reviewed current agents: `49/49`
- Blockers: `need_at_least_2000_reviewed_samples`

## OCR Field Targets

### uid_panel_full_uid

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=1` `val=0` `test=0` `unsplit=0`

### agent_detail_level

- Reviewed records: `506`
- Complete records: `1`
- Splits: `train=0` `val=1` `test=0` `unsplit=0`

### agent_detail_mindscape

- Reviewed records: `506`
- Complete records: `1`
- Splits: `train=0` `val=1` `test=0` `unsplit=0`

### agent_detail_stats

- Reviewed records: `506`
- Complete records: `1`
- Splits: `train=0` `val=1` `test=0` `unsplit=0`

### amplifier_detail

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=1` `val=0` `test=0` `unsplit=0`

### disk_detail

- Reviewed records: `6`
- Complete records: `6`
- Splits: `train=5` `val=1` `test=0` `unsplit=0`

### equipment_overview

- Reviewed records: `481`
- Complete records: `480`
- Splits: `train=378` `val=50` `test=52` `unsplit=0`

### roster_owned_agents

- Reviewed records: `104`
- Complete records: `104`
- Splits: `train=74` `val=15` `test=15` `unsplit=0`
