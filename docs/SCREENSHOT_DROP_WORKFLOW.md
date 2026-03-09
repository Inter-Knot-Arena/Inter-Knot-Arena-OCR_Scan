# Screenshot Drop Workflow

Use this when account-import screenshots already exist as normal PNG/JPG files and you want them appended into the OCR manifest without manual box annotation.

## Supported screen buckets

The importer classifies files by folder names or file names. These aliases are supported:

- `uid`
- `roster`
- `agent_detail`
- `equipment`
- `disk1` ... `disk6`
- `amplificator` or `amplifier`
- `mindscape` (optional provenance only; primary mindscape truth should come from `agent_detail`)

## Recommended folder layout

```text
D:\IKA_DATA\ocr\drops\batch_001\
  uid\
  roster\
  billy_kid\
    agent_detail.png
    equipment.png
    amplificator.png
    disk1.png
    disk4.png
    disk6.png
  lucy_de_montefio\
    agent_detail.png
    equipment.png
    amplificator.png
    disk3.png
    disk5.png
    disk6.png
```

The importer also works if the files are mixed in one folder, as long as the file names contain the bucket name.

Generic names like `Снимок экрана (17).png` are not enough for reliable import. Rename them or sort them into per-agent folders with stable names before importing.

## Empty slots rule

`equipment.png` is the source of truth for what is actually equipped.

- If a disk slot is empty, do not include `diskN.png` for that slot.
- If the amplifier slot is empty, do not include `amplificator.png`.
- If clicking an empty slot opens a recommended item screen, do not use that screen in the dataset.

Only import detail screens for slots that are really occupied on the `equipment.png` overview.

## Import command

```powershell
python scripts/ingest_account_screenshots.py --manifest dataset_manifest.json --input-root D:\IKA_DATA\ocr\drops\batch_001 --locale RU
```

## Ordered generic screenshot series

If screenshots were saved as a stable sequence like:

- `agent_detail`
- `equipment`
- `disk1`
- `disk2`
- `disk3`
- `disk4`
- `disk5`
- `disk6`
- `amplificator`

then you do not need to rename them manually. First materialize them into a structured drop folder:

```powershell
python scripts/materialize_ordered_account_batch.py `
  --input-root C:\Users\loval\OneDrive\Pictures\Screenshots\dataset `
  --output-root D:\IKA_DATA\ocr\drops\batch_20260309_agents_129_255 `
  --start-index 129 `
  --end-index 255 `
  --agent "Е Шуньгуан" `
  --agent "Джейн Доу"
```

Then import that structured batch with the normal screenshot importer.

Use `--skip canonical_agent_id:role1,role2` for empty slots. Example:

```powershell
--skip "agent_billy:disk2,disk3,disk5"
--skip "agent_grace:disk2,amplificator"
```

If a series is shorter for some agents, use `--agent-pack` instead of a single global role order:

```powershell
python scripts/materialize_ordered_account_batch.py `
  --input-root C:\Users\loval\OneDrive\Pictures\Screenshots `
  --output-root D:\IKA_DATA\ocr\drops\batch_20260309_agents_256_268 `
  --start-index 256 `
  --end-index 268 `
  --agent-pack "Антон Иванов:agent_detail,equipment,amplificator" `
  --agent-pack "Чжао:agent_detail,equipment"
```

Optional flags:

- `--batch-id ellen_ru_20260308`
- `--resolution 1080p`
- `--game-patch 2.6`
- `--capture-date 2026-03-08T12:30:00Z`
- `--dry-run`

## What the importer does

1. Detects screen role from the folder path or file name.
2. Copies each image into private storage under `raw\manual_screens\<batch-id>\...`.
3. Appends one `source` and one `record` per screenshot into `dataset_manifest.json`.
4. Marks everything as `workflow=account_import`.
5. Preserves optional `mindscape` captures in the manifest, but keeps them outside the current import-eligible training slice because primary mindscape truth should come from `agent_detail`.

## Roster ownership review

Full-page `roster` screenshots should be reviewed as account ownership truth, not as a fake single `agent_icon_id`.

Example:

```powershell
python scripts/apply_roster_ownership_review.py --manifest dataset_manifest.json --session-id onedrive_account_20260308 --reviewer-id loval --owned-agent "Эллен Джо" --owned-agent "Джейн Доу"
```

The script writes `owned_agent_ids` and `not_owned_agent_ids` into the reviewed roster records and marks those screenshots as `qaStatus=reviewed`.

## After import

Rebuild the OCR artifacts:

```powershell
python scripts/build_account_capture_backlog.py --manifest dataset_manifest.json --output-file docs/account_capture_backlog.json
python scripts/qa_audit.py --manifest dataset_manifest.json --workflow account_import --output-file docs/qa_report.json --double-review-file docs/double_review_samples.json
python scripts/build_review_batches.py --manifest dataset_manifest.json --workflow account_import --status needs_review,unlabeled --output-csv docs/review_queue.priority.csv --output-json docs/review_batches.json
python scripts/export_dataset_preview.py --manifest dataset_manifest.json --workflow account_import --output-csv docs/dataset_preview.csv --output-html docs/dataset_preview.html
python scripts/build_sampling_plan.py --manifest dataset_manifest.json --workflow account_import --target-uid-digit 20000 --target-agent-icon 15000 --target-equipment 15000 --output-file docs/sampling_plan.json
python scripts/build_roster_coverage.py --manifest dataset_manifest.json --workflow account_import --output-file docs/roster_coverage.json
```
