# E9 External Human-Ranked Evaluation: Screening and Locked Protocol

## Decision

The primary E9 dataset is **IBM Argument Quality Ranking 30k**. The experiment
uses only its official test split at the pinned revision
`590726b3765b1b90c5e53a17e3b1f77d92d3aa8a`.

This is the cleanest match to the dissertation task. The dataset contains
30,497 crowd-written arguments for 71 controversial topics. Each argument has
pointwise human quality and stance labels from ten annotators. The official
dataset card and the AAAI 2020 paper describe the collection and label design:

- [Official IBM dataset card](https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k)
- [Gretz et al., 2020 paper](https://arxiv.org/abs/1911.11408)

The source CSV is stored at
`datasets/external/ibm_argument_quality_ranking_30k/test.csv` and has SHA-256
`18c0f0491e0206558f24dbe79b91901bd662e5ec2b1bd9d13ab740660b0c196f`.

## Dataset screening

| Dataset | Human target | Fit to this task | Decision |
|---|---|---|---|
| IBM Argument Quality Ranking 30k | Pointwise speech-use quality; ten annotations per argument; pro/con stance | Directly supports claim-conditioned argument-strength ranking | **Primary E9** |
| UKPConvArgRank | Pairwise convincingness, with global PageRank ranks | Strong pairwise alternative, but smaller: 1,052 arguments across 32 topics; global order is graph-derived | Reserve replication |
| ArgSum / ECR | Evidence convincingness ranking in an end-to-end argument-summarisation benchmark | Recent and interesting, but its released task/data interface is less direct for the present four-argument scoring protocol | Not primary |
| Anthropic Persuasion | Change in one reader's pre/post agreement after reading human- or model-written text | Measures persuasion rather than intrinsic argument quality and includes model-generated arguments | Domain-shift extension only |

Primary sources for the alternatives are the
[UKPConvArgRank ACL paper](https://public.ukp.informatik.tu-darmstadt.de/UKP_Webpage/publications/2016/acl2016-convincing-arguments-camera-ready.pdf),
the [ArgSum ACL 2024 paper](https://arxiv.org/abs/2406.03151), and the
[Anthropic Persuasion dataset card](https://huggingface.co/datasets/Anthropic/persuasion).

The IBM card currently contains a licensing inconsistency: its metadata says
CC-BY-3.0, whereas its licensing section says CC-BY-SA-3.0. Reporting and reuse
must follow the stricter **CC-BY-SA-3.0** interpretation unless clarified by
the dataset owner.

## Locked preprocessing

The preprocessing is deterministic and contains no manual example selection.

1. Use only rows whose `set` is `test`.
2. Keep pro/con rows with `stance_WA` in `{-1, +1}`.
3. Require `stance_WA_conf >= 0.80`.
4. Use `WA`, the weighted-average human quality label, as the primary order.
   Retain `MACE-P` only as provenance and a deterministic tie-breaker when
   choosing one representative text per exact WA value.
5. Select two pilot topics by the lowest SHA-256 values under the fixed
   namespace `e9-pilot-topic-v1`; do not select them by content or difficulty.
6. Exclude both stances and both panels of those pilot topics from formal E9.

After filtering, 5,818 source arguments remain across all 15 official test
topics. The two excluded pilot topics are:

- `Social media brings more harm than good`
- `We should ban algorithmic trading`

Formal E9 contains the remaining 13 topics, 26 topic-by-stance source groups,
and two four-argument panels per group:

- **Broad panel (primary):** one argument near each locked within-group quality
  quantile: `.10`, `.3667`, `.6333`, and `.90`. The final order is descending
  WA. Its mean rank-1-to-rank-4 WA span is `0.4191`.
- **Near-tie panel (secondary challenge):** four distinct WA values from the
  central 40--60% of the group, choosing the window with the smallest WA span
  after excluding broad-panel items. Its mean total WA span is only `0.00166`.

The near-tie panel is not a second strong-gold accuracy benchmark. Such tiny
human-label gaps may be practically indistinguishable. It tests whether score
dispersion, ties, and preference instability increase when the reference
ordering is intrinsically weak.

Prepared and locked artefacts:

- `configs/e9_external.yaml`
- `configs/splits/e9_external_manifest.json`
- `datasets/external/ibm_argument_quality_ranking_30k/e9_pilot.jsonl`
- `datasets/external/ibm_argument_quality_ranking_30k/e9_formal.jsonl`
- `results/uq_apo/candidates/e9_pilot_candidates.jsonl`
- `results/uq_apo/candidates/e9_five_prompts.jsonl`

The formal JSONL SHA-256 is
`e77fd9fb9eb790ad993c8931e48733046b63f14fe934c85a885bc95600edda4e`.

## Excluded format pilot

The pilot evaluates only Direct and the frozen winner on eight four-argument
records, using seeds 42 and 1024: 128 score generations in total.

The gate checks only:

- exact row completeness;
- locked prompt, seed, topic, and stage identity;
- 100% parse success;
- 100% exact hard-decoding format compliance;
- score range 0--100.

Pilot accuracy is deliberately not calculated. The formal job starts only if
this format/completeness gate returns success.

## Frozen formal experiment

Formal E9 evaluates these five readable prompts:

| Prompt | ID |
|---|---|
| Direct | `p_83adf7318fd5` |
| Role | `p_da4eef59b8ac` |
| Rubric | `p_0086c8edb3b1` |
| Role+Rubric | `p_26918ada0c0a` |
| Frozen APO winner | `p_53a4acbe525c` |

The task model is Qwen3-8B. Generation is Hard, No-Think, temperature `.7`,
top-p `.95`, repetition penalty `1.0`, maximum 12 new tokens, with the same ten
seeds used in validation and internal testing. The design therefore requires:

`52 records x 4 arguments x 5 prompts x 10 seeds = 10,400 generations`.

No external result may alter the winner, prompt text, panel construction, or
hyperparameters.

## Confirmatory performance analysis

The independent inference unit is the original source topic (`n=13`), not a
panel record, stance group, argument pair, seed, or one of the 252 UQ
partitions.

For each prompt, aggregate accuracy, seed flip, score variance, invalid rate,
tokens, and latency within each topic. Compare winner minus each manual prompt
with a 10,000-resample topic-cluster bootstrap CI and a 100,000-resample paired
sign-flip test. Apply Holm correction within panel and metric.

- Primary panel: `broad`.
- Secondary panel: `near_tie`.
- Pooled `all` is descriptive because it mixes strong- and weak-order panels.

## UQ transfer analysis

Ten runs are divided into every disjoint 5+5 estimate/held-out split,
producing 252 repeated views. These are not 252 independent datasets.

The two frozen primary signals follow the internal validation findings:

- `sampling_variance` predicts future absolute score/margin deviation;
- `margin_uncertainty` predicts future reference-order error.

For calibration transfer, an isotonic mapping from `margin_uncertainty` to
future error is fitted on the internal Qwen validation claims for each seed
partition and applied unchanged to external E9. External data are not used to
refit this primary calibrator. Report external AUROC/AUPRC, validation-fitted
Brier/ECE, Spearman, and risk-coverage results separately for broad and
near-tie panels.

## Execution chain

Submitted jobs on 4 August 2026:

- `270014` (A30): excluded pilot plus non-accuracy gate;
- `270016` (A40): formal ten-seed E9, dependent on successful `270014`;
- `270017`: external performance and UQ-transfer analysis, dependent on
  successful `270016`.


Job `270014` completed successfully: all 128 expected rows were present and
both parse success and exact hard-format compliance were 100%.
Manual commands, if a job must be reproduced, are:

```bash
python scripts/prepare_e9_external.py
sbatch slurm/run_e9_external.sbatch pilot
sbatch --dependency=afterok:PILOT_JOB_ID slurm/run_e9_external.sbatch formal
python scripts/analyse_e9_external.py
```

## 中文摘要

E9 选择 IBM-Rank-30k，是因为它真正提供了独立的、人类标注的 argument
quality 与 stance，而不是由原 149 claims 改写得到的数据。正式实验保留 13 个
topic，并以 topic 为统计独立单位。Broad panel 是主要外部准确率测试；near-tie
panel 的人类分数差极小，只用于检验模型在弱 gold ordering 下是否出现更高方差、
更多 tie 和更多 seed flip。UQ calibration 只在内部 validation 上拟合，然后原样
迁移到 E9，不能用外部结果重新校准或改写 frozen winner。
