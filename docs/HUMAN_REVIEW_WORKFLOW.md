# Human Review Workflow

## Goal
Turn `suggestedLabels` into verified `labels` before any CUDA training run.

## Principles
- `suggestedLabels` are model guesses only.
- `labels` are ground truth only after human review.
- Do not train production models on unreviewed rows.

## CV
1. Build prioritized batches:
```powershell
python scripts/build_review_batches.py --manifest dataset_manifest.json --output-csv docs/review_queue.priority.csv --output-json docs/review_batches.json
```
2. Review rows in `docs/review_queue.priority.csv`.
3. Apply round A/B or final labels:
```powershell
python scripts/apply_review_labels.py --manifest dataset_manifest.json --input-csv docs/review_queue.priority.csv --review-round final --reviewer-id loval
```
4. Refresh QA:
```powershell
python scripts/qa_audit.py --manifest dataset_manifest.json --output-file docs/qa_report.json --double-review-file docs/double_review_samples.json
```

## OCR
1. Build prioritized batches:
```powershell
python scripts/build_review_batches.py --manifest dataset_manifest.json --workflow account_import --output-csv docs/review_queue.priority.csv --output-json docs/review_batches.json
```
2. Review rows in `docs/review_queue.priority.csv`.
3. Apply round A/B or final labels:
```powershell
python scripts/apply_review_labels.py --manifest dataset_manifest.json --input-csv docs/review_queue.priority.csv --review-round final --reviewer-id loval
```
4. Refresh QA:
```powershell
python scripts/qa_audit.py --manifest dataset_manifest.json --workflow account_import --output-file docs/qa_report.json --double-review-file docs/double_review_samples.json
```
5. Do not review `combat_reference` rows in OCR queues. Those belong to match-monitor/CV context, not roster import.

## Validation
- CV agent ids must resolve to canonical roster ids or `unknown`.
- OCR `uid_digit` must be a single digit `0-9`.
- OCR `agent_icon_id` must resolve to a canonical roster id or `unknown`.
- OCR `disc_level` must be numeric or `unknown`.

## Current priority
- Review records for agents missing in `reviewed` counts.
- Review low-confidence and `unknown_flag` rows first.
- Review `uid_digit` rows early because UID OCR is currently the weakest head.
- Prioritize `uid_panel`, `roster`, `agent_detail`, and `equipment` screen roles only.
