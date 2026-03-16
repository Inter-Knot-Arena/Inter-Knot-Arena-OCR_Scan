# OCR Training Readiness

- Generated at: `2026-03-16T04:31:00Z`
- Workflow: `account_import`
- Scoped records: `8804`
- Reviewed truth records: `8466`

## Current Train Heads

### uid_digit (READY)

- Samples: `2230`
- Distinct labels: `10`
- Splits: `train=1850` `val=180` `test=200` `unsplit=0`
- Missing digits: `none`
- Materialized digit samples: `2230`

### agent_icon (READY)

- Samples: `2480`
- Distinct labels: `49`
- Splits: `train=2024` `val=235` `test=221` `unsplit=0`
- Reviewed current agents: `49/49`

### disk_detail (READY)

- Samples: `2497`
- Distinct labels: `24`
- Splits: `train=1978` `val=255` `test=264` `unsplit=0`
- Reviewed disc-set labels: `24`

## OCR Field Targets

### uid_panel_full_uid

- Reviewed records: `223`
- Complete records: `223`
- Splits: `train=185` `val=18` `test=20` `unsplit=0`

### agent_detail_level

- Reviewed records: `1518`
- Complete records: `322`
- Splits: `train=254` `val=36` `test=32` `unsplit=0`

### agent_detail_mindscape

- Reviewed records: `1518`
- Complete records: `106`
- Splits: `train=82` `val=12` `test=12` `unsplit=0`

### agent_detail_stats

- Reviewed records: `1518`
- Complete records: `143`
- Splits: `train=112` `val=17` `test=14` `unsplit=0`

### amplifier_detail

- Reviewed records: `451`
- Complete records: `451`
- Splits: `train=358` `val=48` `test=45` `unsplit=0`

### disk_detail

- Reviewed records: `2497`
- Complete records: `2497`
- Splits: `train=1978` `val=255` `test=264` `unsplit=0`

### equipment_overview

- Reviewed records: `1443`
- Complete records: `480`
- Splits: `train=397` `val=38` `test=45` `unsplit=0`

### roster_owned_agents

- Reviewed records: `104`
- Complete records: `104`
- Splits: `train=82` `val=14` `test=8` `unsplit=0`
