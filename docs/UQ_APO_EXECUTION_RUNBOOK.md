# UQ/APO Confirmatory Execution Runbook

This runbook is the operational companion to `EXPERIMENT_INVENTORY_UQ_APO.md`.
It enforces the confirmatory order: optimise on APO-train, select on
APO-validation, freeze once, and only then inspect the internal test.

## 1. Qwen automatic search (currently submitted as job 269428)

```bash
sbatch --export=NIL slurm/run_prompt_optimization.sbatch
```

Expected terminal artefact:
`results/uq_apo/search/candidates_for_promotion.jsonl`.

## 2. Qwen promotion (30 train claims, five seeds)

```bash
sbatch --export=NIL slurm/run_apo_stage.sbatch \
  promotion \
  results/uq_apo/search/candidates_for_promotion.jsonl \
  results/uq_apo/raw/promotion.jsonl

python scripts/analyse_apo_stage.py results/uq_apo/raw/promotion.jsonl \
  --output-dir results/uq_apo/processed/promotion --keep 2

python scripts/select_prompt_candidates.py \
  --candidates results/uq_apo/search/candidates_for_promotion.jsonl \
  --selection results/uq_apo/processed/promotion/selected_prompt_ids.json \
  --output results/uq_apo/candidates/promotion_survivors.jsonl
```

## 3. Qwen validation and 5+5 UQ (29 validation claims, ten seeds)

```bash
sbatch --export=NIL slurm/run_apo_stage.sbatch \
  validation \
  results/uq_apo/candidates/promotion_survivors.jsonl \
  results/uq_apo/raw/validation.jsonl

python scripts/analyse_apo_stage.py results/uq_apo/raw/validation.jsonl \
  --output-dir results/uq_apo/processed/validation_metrics --keep 2

python scripts/analyse_uq.py results/uq_apo/raw/validation.jsonl \
  --stage validation --output-dir results/uq_apo/processed/validation_uq

python scripts/select_final_prompt.py \
  --candidate-summary results/uq_apo/processed/validation_metrics/candidate_summary.csv \
  --pair-uq-summary results/uq_apo/processed/validation_uq/pair_uq_summary.csv \
  --uq-signal sampling_variance \
  --output-dir results/uq_apo/processed/final_selection
```

## 4. Freeze exactly one winner

```bash
python scripts/freeze_final_prompt.py \
  --candidates results/uq_apo/candidates/promotion_survivors.jsonl \
  --selection results/uq_apo/processed/final_selection/selected_prompt_ids.json
```

This creates:

- `results/uq_apo/frozen/qwen_frozen_winner.jsonl`;
- `results/uq_apo/frozen/mistral_transfer_prompts.jsonl`;
- `results/uq_apo/frozen/freeze_manifest.json`.

The manifest records SHA-256 hashes. Post-validation runners reject modified or
unregistered prompt files.

## 5. Qwen internal test (50 claims, ten seeds, one frozen prompt)

```bash
sbatch --export=NIL slurm/run_frozen_evaluation.sbatch \
  qwen_internal_test \
  results/uq_apo/frozen/qwen_frozen_winner.jsonl \
  results/uq_apo/raw/qwen_internal_test.jsonl

python scripts/analyse_apo_stage.py results/uq_apo/raw/qwen_internal_test.jsonl \
  --output-dir results/uq_apo/processed/qwen_internal_test --keep 1

python scripts/analyse_uq.py results/uq_apo/raw/qwen_internal_test.jsonl \
  --stage internal_test --output-dir results/uq_apo/processed/qwen_internal_test_uq
```

Do not rewrite, reselect, or retune after observing this result.

## 6. Mistral frozen-prompt transfer

The transfer registry contains Direct, Rubric, and the frozen Qwen winner. If
the winner is itself Direct or Rubric, duplicate prompt IDs are collapsed.

```bash
sbatch --export=NIL slurm/run_frozen_evaluation.sbatch \
  cross_model_transfer \
  results/uq_apo/frozen/mistral_transfer_prompts.jsonl \
  results/uq_apo/raw/mistral_transfer.jsonl

python scripts/analyse_apo_stage.py results/uq_apo/raw/mistral_transfer.jsonl \
  --output-dir results/uq_apo/processed/mistral_transfer --keep 3
```

Primary transfer claim: whether a prompt optimised only from Qwen development
data improves or preserves accuracy and stability on Mistral relative to the
two frozen manual baselines. This stage must not trigger any further prompt
rewriting.

## Monitoring and recovery

```bash
squeue -u xz3725
tail -f slurm-uq-apo-search-269428.out
tail -f slurm-uq-apo-search-269428.err
```

All generation writers are resumable: resubmitting the same command skips
completed model/prompt/seed/claim/argument cells. Never delete a partial raw
JSONL merely because a Slurm time limit or node failure interrupted the job.
