# OCR Training Readiness

- Generated at: `2026-03-10T14:31:33Z`
- Workflow: `account_import`
- Scoped records: `134`
- Reviewed truth records: `15`

## Current Train Heads

### uid_digit (BLOCKED)

- Samples: `0`
- Distinct labels: `0`
- Splits: `train=0` `val=0` `test=0` `unsplit=0`
- Missing digits: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Blockers: `need_at_least_2000_reviewed_samples; need_at_least_10_distinct_labels; missing_train_split_samples; missing_val_split_samples; missing_test_split_samples`

### agent_icon (BLOCKED)

- Samples: `2`
- Distinct labels: `1`
- Splits: `train=0` `val=0` `test=0` `unsplit=2`
- Reviewed current agents: `1/49`
- Missing reviewed agents: `agent_anby, agent_nicole, agent_billy, agent_soldier_11, agent_nekomiya_mana, agent_corin, agent_lycaon, agent_koleda, agent_anton, agent_ben, agent_soukaku, agent_miyabi, agent_grace, agent_rina, agent_zhu_yuan, agent_lucy, agent_piper, agent_qingyi, agent_jane_doe, agent_seth, agent_caesar, agent_burnice, agent_yanagi, agent_lighter, agent_harumasa, agent_astra_yao, agent_evelyn, agent_soldier_0_anby, agent_pulchra, agent_trigger, agent_vivian, agent_hugo, agent_yixuan, agent_pan_yinhu, agent_ju_fufu, agent_yuzuha, agent_alice, agent_seed, agent_orphie_magus, agent_lucia, agent_komano_manato, agent_yidhari, agent_dialyn, agent_banyue, agent_zhao, agent_ye_shunguang, agent_sunna, agent_aria`
- Blockers: `need_at_least_2000_reviewed_samples; need_at_least_2_distinct_labels; missing_train_split_samples; missing_val_split_samples; missing_test_split_samples; missing_current_roster_agent_coverage`

## OCR Field Targets

### uid_panel_full_uid

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=0` `val=0` `test=0` `unsplit=1`

### agent_detail_level

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=0` `val=0` `test=0` `unsplit=1`

### agent_detail_mindscape

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=0` `val=0` `test=0` `unsplit=1`

### agent_detail_stats

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=0` `val=0` `test=0` `unsplit=1`

### amplifier_detail

- Reviewed records: `1`
- Complete records: `1`
- Splits: `train=0` `val=0` `test=0` `unsplit=1`

### disk_detail

- Reviewed records: `6`
- Complete records: `6`
- Splits: `train=0` `val=0` `test=0` `unsplit=6`

### equipment_overview

- Reviewed records: `1`
- Complete records: `0`
- Splits: `train=0` `val=0` `test=0` `unsplit=0`

### roster_owned_agents

- Reviewed records: `5`
- Complete records: `5`
- Splits: `train=0` `val=0` `test=0` `unsplit=5`
