# Dissertation Experiment Inventory: Reliability, UQ, and Automatic Prompt Optimisation

This document separates completed experiments from the new prospectively locked
UQ/APO pipeline. It is intended as the master checklist for implementation and
reporting.

## Research narrative

1. **Measure reliability.** Main, Decoding, and Reasoning studies test whether
   argument-strength scores are accurate, stable, and protocol-sensitive.
2. **Identify uncertainty.** Repeated-score UQ tests which estimate-set signals
   predict errors and instability on disjoint future generations.
3. **Mitigate uncertainty.** UQ-guided, embedding-assisted discrete search
   automatically proposes and selects readable prompts.
4. **Validate mitigation.** The selected prompt is frozen, evaluated on a
   claim-disjoint internal test split, and then tested on external data.

This is **Uncertainty Quantification** because uncertainty signals are defined,
evaluated against future outcomes, and calibrated. It is **automatic prompt
optimisation** because evaluation, failure selection, critique, rewriting,
deduplication, and selection form a repeated algorithmic loop. It is
**embedding-assisted discrete optimisation**, not soft-prompt optimisation:
the prompts remain readable text, while embeddings only organise examples and
candidates.

## Completed studies

| ID | Study | Design | Primary question | Main result | Status |
|---|---|---|---|---|---|
| E0 | Output-length pilots | 10 claims; Qwen/GPT-OSS; soft; Think/No-Think | Are token ceilings adequate? | Established mode-specific generation limits and truncation diagnostics. | Completed |
| E1 | Decoding Validation | 20 claims (10 support, 10 attack); 3 models; Rubric; Hard vs Soft; No-Think; 3 seeds | Does constrained output change measurement? | Hard decoding guarantees exact format and is much cheaper, but effects are model-dependent. Qwen was nearly identical; GPT-OSS hard decoding had lower accuracy and more flips in this pilot. | Completed |
| E2 | Main Study | 149 claims; 4 arguments each; 3 models; Direct/Role/Rubric/Role+Rubric; Hard; No-Think; 3 seeds | Do model and manual prompt design change ranking quality? | Pairwise accuracy was high (roughly .904--.931), but no prompt delivered a stable cross-model improvement. Exact-order accuracy was much lower. | Completed |
| E3 | Reasoning Study | 50 claims (25/25); Qwen and GPT-OSS; 4 prompts; Soft; Think vs No-Think; 3 seeds | Does visible reasoning improve judgement? | Qwen Thinking reduced ties and showed a small accuracy trend, but increased repeated-seed preference instability; GPT-OSS had no reliable aggregate gain. | Completed |
| E4 | Representation Study | Expanded/synonym dataset; 3 models; Rubric; Hard; No-Think; 3 seeds | Are scores robust to argument representation? | Configured but no raw result file is currently present. It tests representation sensitivity, not external ranking accuracy. | Not yet run |

### Main Study numbers to report

Seed-averaged pairwise accuracy by prompt:

| Model | Direct | Role | Rubric | Role+Rubric |
|---|---:|---:|---:|---:|
| GPT-OSS-20B | .910 | .905 | .919 | .931 |
| Mistral-7B | .908 | .914 | .918 | .904 |
| Qwen3-8B | .919 | .925 | .930 | .931 |

These means show local trends, not a universal prompt ordering. The formal
prompt contrasts and Holm-adjusted tests are in
`results/processed/main_study/prompt_comparisons.csv`. The equivalence tests do
not establish that every complex prompt is practically equivalent to Direct;
the correct conclusion is that the experiment found neither robust improvement
nor sufficiently precise evidence of equivalence.

### Decoding Validation numbers to report

| Model | Hard accuracy | Soft accuracy | Hard flip | Soft flip | Hard tokens | Soft tokens |
|---|---:|---:|---:|---:|---:|---:|
| GPT-OSS-20B | .908 | .939 | .258 | .132 | 5.0 | 228.9 |
| Mistral-7B | .917 | .925 | .192 | .108 | 5.8 | 116.3 |
| Qwen3-8B | .908 | .908 | .092 | .092 | 6.0 | 6.0 |

The pilot is small, so it supports a protocol-sensitivity warning rather than
a universal claim that Soft is better. Mistral Soft had 100% score extraction
but only 7.9% exact-format compliance; Hard had 100% exact compliance.

### Reasoning Study results to report

- GPT-OSS: averaged over prompts, Thinking changed pairwise accuracy by about
  +0.007; the Holm-adjusted test was not significant.
- Qwen: averaged over prompts, Thinking changed pairwise accuracy by about
  +0.020; the Holm-adjusted test was not significant.
- Qwen tie rate fell by about 0.034 overall and this survived Holm correction.
- In adjacent pairs, Qwen accuracy rose from about .859 to .877 and tie rate
  fell from .085 to .040, while seed flip rose from .135 to .275.
- Of 66 Qwen No-Think tied pairs, 49 moved to the correct direction, 9 to the
  wrong direction, and 8 remained tied; 39/66 disagreed across Thinking seeds.

Interpretation: visible reasoning often makes Qwen more decisive, but
decisiveness is not the same as epistemic certainty or repeatability.

## Completed secondary analyses (A--F)

| ID | Analysis | Output | Finding |
|---|---|---|---|
| A | Rank-distance analysis | `rank_distance_*.csv/png` | Errors are concentrated among adjacent arguments; distance-3 accuracy is approximately .98--.99. |
| B | Qwen tie transitions | `qwen_tie_transition*.csv/png` | Thinking usually resolves mean ties correctly but frequently disagrees across seeds. |
| C | Score-scale analysis | `score_*.csv/png`, `pair_margin_analysis.csv` | Models discretise and spread the 0--100 scale differently, limiting cross-model interpretation of a fixed 10-point strict margin. |
| D | Claim difficulty/error audit | `claim_difficulty_ranking.csv`, `claim_error_audit.csv`, `claim_pair_disagreements.csv` | Difficult and reference-questionable cases are concentrated and available for manual coding. |
| E | Accuracy-stability-cost Pareto analysis | `tradeoff_cells.csv`, `accuracy_stability_cost.png` | Accuracy alone can hide instability and orders-of-magnitude reasoning cost. |
| F | Equivalence testing | `equivalence_tests.csv`, `prompt_equivalence_forest.png` | No prompt contrast met the predefined equivalence bounds after within-model Holm correction. |

All files above are under `results/processed/extended_analysis/` or
`results/figures/extended_analysis/`.

## New UQ/APO protocol

### E5. Locked development design

- Dataset: the existing 149-claim ranking dataset.
- Claim-disjoint partitions: 70 APO-train, 29 APO-validation, 50 internal-test.
- Stance allocation: 39/31, 16/13, and 27/23 support/attack respectively.
- Split file contains a dataset SHA-256 fingerprint and is never regenerated
  after optimisation starts.
- Limitation: earlier studies inspected all 149 claims. Therefore the new
  internal test is held out from the algorithm, but is not a pristine external
  test from the perspective of the full dissertation process.

### E6. UQ estimator study

**Primary model/protocol:** Qwen3-8B, Rubric or candidate prompt, Hard,
No-Think, temperature .7, top-p .95.

For every prompt/claim/argument, generate 10 stochastic runs. Every five-run
estimate set is paired with the disjoint five-run complement, yielding all
`C(10,5)=252` partitions.

Argument-level signals:

- sampling variance;
- sampling range;
- median absolute deviation;
- score-option entropy.

Pair-level signals add:

- preference entropy over first-preferred/tie/second-preferred outcomes;
- pair disagreement probability;
- inverse absolute aggregate margin (`margin_uncertainty`).

Held-out targets:

- future reference-order error;
- future pairwise loss;
- future absolute score/margin deviation;
- future pair disagreement;
- future flip from the estimate-set decision.

Evaluation metrics:

- Spearman correlation with future deviation;
- AUROC and AUPRC for future error;
- risk-coverage/AURC;
- claim-grouped cross-fitted isotonic Brier score and ECE.

The 252 partitions are repeated views of the same claims and must not be
reported as 252 independent datasets.

### E7. UQ-aware automatic prompt search

The frozen components are the task definition, Claim/Role/Argument input
fields, 0--100 scale, exact output format, target model, and decoding settings.
Only the readable instruction/rubric block is optimised.

One search round is:

1. Evaluate each candidate on a rotating balanced set of 10 APO-training
   claims using **3 seeds** (120 score generations per candidate).
2. Calculate pairwise accuracy, normalised score variance, seed flip rate,
   invalid rate, tokens, and latency.
3. Prioritise wrong, tied, adjacent, high-dispersion, and seed-flipping pairs.
4. Embed failure descriptions and select medoids from distinct clusters.
5. Give the parent prompt and representative failures to GPT-OSS-20B for
   structured critique and readable rewrites.
6. Embed candidate instructions and remove semantic near-duplicates.
7. Retain the Pareto set and rank it with an accuracy lower-confidence bound
   penalised by variance, flips, invalidity, calibration error, and cost.
8. Repeat on a new rotating APO-training mini-batch.

Implementation note: during screening, candidate Brier values were unavailable
and shared the same fallback, while all completions were below the scalar
50-token penalty threshold. Calibration therefore became active only at
validation; generated tokens and latency affected screening through Pareto
dominance, not through a non-zero scalar token penalty.

Three seeds are preferable to two at this screening stage: two runs can only
be equal or different, while three expose unanimous versus 2--1 outcomes.
Three runs are still too few for the dissertation's final UQ claim, which uses
5 estimate plus 5 held-out runs.

### E8. Successive halving and final evaluation

| Stage | Claims | Seeds | Use |
|---|---:|---:|---|
| Search screening | 10 rotating training claims | 3 | Cheap candidate rejection and failure discovery |
| Promotion | 30 training claims | 5 | Reduce search survivors to two |
| Full training assessment | 70 training claims | 5 | Check search objective over all optimisation claims |
| Validation | 29 validation claims | 10 (5+5) | Compare UQ, calibration, stability, cost; select one prompt |
| Internal test | 50 test claims | 10 (5+5) | Evaluate the frozen winner once; no further rewriting |
| External ranking test | 13 independent IBM test topics; 52 four-argument records | 10 (5+5) | Test genuine generalisation and UQ transfer |
| Representation robustness | Expanded/synonym derived data | 3--10 | Test invariance, not external accuracy |

### E9. Locked external human-ranked test

E9 now uses the official test split of IBM Argument Quality Ranking 30k at
revision `590726b3765b1b90c5e53a17e3b1f77d92d3aa8a`. Two deterministically
hashed pilot topics are fully excluded. The remaining 13 source topics produce
26 topic-by-stance groups and two panels per group: a broad-quality primary
panel and a near-tie uncertainty challenge panel.

The excluded Direct-versus-winner pilot contains 128 generations and is gated
only on completeness and exact-format parsing. The frozen formal design is 52
records x 4 arguments x 5 prompts x 10 seeds = 10,400 generations. Inference
is clustered by the 13 original topics. External results cannot change the
frozen winner.

For UQ transfer, `sampling_variance` predicts future absolute deviation and
`margin_uncertainty` predicts future reference error. The primary isotonic
calibrator is fitted on internal Qwen validation and applied externally without
refitting. See `docs/E9_EXTERNAL_DATASET_SCREENING_AND_PROTOCOL.md` and
`configs/e9_external.yaml` for the full lock.

## Commands

Prepare the locked split and four seed candidates once:

```bash
python scripts/prepare_uq_apo.py
```

Run the automatic search:

```bash
python scripts/run_prompt_optimization.py
```

Promote and analyse candidates:

```bash
python scripts/run_apo_stage.py \
  --candidates results/uq_apo/search/candidates_for_promotion.jsonl \
  --stage promotion --output results/uq_apo/raw/promotion.jsonl

python scripts/analyse_apo_stage.py results/uq_apo/raw/promotion.jsonl \
  --output-dir results/uq_apo/processed/promotion --keep 2

python scripts/select_prompt_candidates.py \
  --candidates results/uq_apo/search/candidates_for_promotion.jsonl \
  --selection results/uq_apo/processed/promotion/selected_prompt_ids.json \
  --output results/uq_apo/candidates/promotion_survivors.jsonl
```

Run 10-seed validation and UQ analysis:

```bash
python scripts/run_apo_stage.py \
  --candidates results/uq_apo/candidates/promotion_survivors.jsonl \
  --stage validation --output results/uq_apo/raw/validation.jsonl

python scripts/analyse_uq.py results/uq_apo/raw/validation.jsonl \
  --stage validation --output-dir results/uq_apo/processed/validation_uq

python scripts/analyse_apo_stage.py results/uq_apo/raw/validation.jsonl \
  --output-dir results/uq_apo/processed/validation_metrics --keep 2

python scripts/select_final_prompt.py \
  --candidate-summary results/uq_apo/processed/validation_metrics/candidate_summary.csv \
  --pair-uq-summary results/uq_apo/processed/validation_uq/pair_uq_summary.csv \
  --uq-signal sampling_variance \
  --output-dir results/uq_apo/processed/final_selection
```

After selecting exactly one prompt, freeze its registry and run the internal
test once using `--stage internal_test`. Do not return to prompt rewriting after
seeing those results.

Prepare and run locked E9:

```bash
python scripts/prepare_e9_external.py
sbatch slurm/run_e9_external.sbatch pilot
sbatch --dependency=afterok:PILOT_JOB_ID slurm/run_e9_external.sbatch formal
python scripts/analyse_e9_external.py
```

Run post-freeze E10 adaptive sampling and the independently locked E11 Webis
replication:

```bash
python scripts/analyse_adaptive_uq.py
python scripts/analyse_bayesian_adaptive.py

python scripts/prepare_e11_webis.py
sbatch slurm/run_e11_webis.sbatch pilot
sbatch --dependency=afterok:PILOT_JOB_ID slurm/run_e11_webis.sbatch formal
sbatch --dependency=afterok:FORMAL_JOB_ID slurm/process_e11_webis.sbatch
```

E10 is explicitly post-freeze exploratory. E11 uses two hash-selected topics
only for format validation and excludes them from the formal 18-topic panel.
Its primary gold outcome is the decisive majority of direct human logical,
rhetorical, and dialectical comparisons. See `docs/ADAPTIVE_UQ_EXPERIMENT.md`
and `docs/E11_WEBIS_EXTERNAL_PROTOCOL.md`.

## New files and generated artefacts

Source and configuration:

- `configs/uq_apo.yaml`
- `src/argstrength/experimental_design.py`
- `src/argstrength/uq.py`
- `src/argstrength/embedding.py`
- `src/argstrength/apo.py`
- `src/argstrength/apo_experiment.py`
- `src/argstrength/optimizer_model.py`
- `src/argstrength/tabular.py`
- `scripts/prepare_uq_apo.py`
- `scripts/run_prompt_optimization.py`
- `scripts/run_apo_stage.py`
- `scripts/analyse_apo_stage.py`
- `scripts/analyse_uq.py`
- `scripts/select_final_prompt.py`
- `slurm/run_prompt_optimization.sbatch`
- `slurm/run_apo_stage.sbatch`
- `scripts/select_prompt_candidates.py`
- `tests/test_uq.py`, `tests/test_apo.py`, `tests/test_experimental_design.py`

- `configs/e9_external.yaml`
- `src/argstrength/external_data.py`
- `src/argstrength/external_analysis.py`
- `scripts/prepare_e9_external.py`
- `scripts/run_e9_external.py`
- `configs/splits/e9_external_manifest.json`
- `datasets/external/ibm_argument_quality_ranking_30k/e9_formal.jsonl`
- `results/uq_apo/candidates/e9_five_prompts.jsonl`
- `scripts/validate_e9_pilot.py`
- `scripts/analyse_e9_external.py`
- `slurm/run_e9_external.sbatch`
- `slurm/process_e9_external.sbatch`
- `tests/test_external_data.py`, `tests/test_external_analysis.py`
- `configs/adaptive_uq.yaml`, `configs/bayesian_adaptive.yaml`
- `src/argstrength/adaptive_uq.py`, `src/argstrength/bayesian_adaptive.py`
- `scripts/analyse_adaptive_uq.py`, `scripts/analyse_bayesian_adaptive.py`
- `tests/test_adaptive_uq.py`, `tests/test_bayesian_adaptive.py`
- `configs/e11_webis_external.yaml`
- `src/argstrength/webis_external.py`, `src/argstrength/webis_analysis.py`
- `scripts/prepare_e11_webis.py`, `scripts/run_e11_webis.py`
- `scripts/validate_e11_pilot.py`, `scripts/analyse_e11_webis.py`
- `slurm/run_e11_webis.sbatch`, `slurm/process_e11_webis.sbatch`
- `tests/test_webis_external.py`, `tests/test_webis_analysis.py`
Already generated by preparation:

- `configs/splits/uq_apo_claim_splits.json`
- `results/uq_apo/candidates/iteration_0.jsonl`

Future generated results are checkpointed under `results/uq_apo/raw/`, with
derived tables and manifests under `results/uq_apo/processed/` and
`results/uq_apo/search/`.
