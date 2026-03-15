# OCR Training Readiness

- Generated at: `2026-03-15T23:01:01Z`
- Workflow: `account_import`
- Scoped records: `7454`
- Reviewed truth records: `6981`

## Current Train Heads

### uid_digit (BLOCKED)

- Samples: `880`
- Distinct labels: `10`
- Splits: `train=690` `val=110` `test=80` `unsplit=0`
- Missing digits: `none`
- Materialized digit samples: `880`
- Blockers: `need_at_least_2000_reviewed_samples`

### agent_icon (READY)

- Samples: `2480`
- Distinct labels: `49`
- Splits: `train=1979` `val=218` `test=283` `unsplit=0`
- Reviewed current agents: `49/49`

## OCR Field Targets

### uid_panel_full_uid

- Reviewed records: `88`
- Complete records: `88`
- Splits: `train=69` `val=11` `test=8` `unsplit=0`

### agent_detail_level

- Reviewed records: `1518`
- Complete records: `322`
- Splits: `train=251` `val=30` `test=41` `unsplit=0`

### agent_detail_mindscape

- Reviewed records: `1518`
- Complete records: `106`
- Splits: `train=83` `val=5` `test=18` `unsplit=0`

### agent_detail_stats

- Reviewed records: `1518`
- Complete records: `143`
- Splits: `train=104` `val=20` `test=19` `unsplit=0`

### amplifier_detail

- Reviewed records: `451`
- Complete records: `451`
- Splits: `train=364` `val=38` `test=49` `unsplit=0`

### disk_detail

- Reviewed records: `2497`
- Complete records: `2497`
- Splits: `train=1987` `val=272` `test=238` `unsplit=0`

### equipment_overview

- Reviewed records: `1443`
- Complete records: `480`
- Splits: `train=394` `val=37` `test=49` `unsplit=0`

### roster_owned_agents

- Reviewed records: `104`
- Complete records: `104`
- Splits: `train=90` `val=6` `test=8` `unsplit=0`
