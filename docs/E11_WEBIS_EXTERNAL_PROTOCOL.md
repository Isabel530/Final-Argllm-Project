# E11: Frozen External Validation on Direct Human Pairwise Judgements

## Purpose

E11 tests whether the already-frozen Qwen APO prompt and its uncertainty
behaviour generalise to a second, previously unused argument-quality corpus.
It complements E9 rather than replacing it. E9 uses IBM scalar quality labels;
E11 uses the raw human pair comparisons in Webis-ArgQuality-20.

The corpus contains 1,271 valid arguments from 20 topics. Its logical,
rhetorical, and dialectical quality scores were collected through 41,859
pairwise judgements. Source: Gienapp et al. (ACL 2020),
<https://aclanthology.org/2020.acl-main.511/>; archived dataset version 1.0.0,
<https://doi.org/10.5281/zenodo.3780049> (CC-BY-4.0).

## Frozen compatibility filters

The existing judge assigns strength relative to one claim and one stated role.
Consequently, E11 includes a raw human comparison only when:

1. both endpoints are marked as arguments;
2. both belong to the same Webis topic;
3. both have the same pro/con stance.

Cross-stance comparisons are excluded because one shared `Role:` field cannot
truthfully describe both endpoints. This rule was fixed before model
generation, not selected from model results.

The primary gold label is the majority of the three direct human dimension
labels. The primary analysis retains decisive A/B majorities. Prespecified
sensitivity analyses use unanimous decisive labels, tie-aware majority labels,
each quality dimension separately, and the corpus's inferred Combined Quality.

## Pilot and formal split

Two complete topics are selected by a deterministic hash for a plumbing-only
pilot. They are excluded from the formal panel. The remaining 18 topics are the
independent clusters for confidence intervals and paired tests. Arguments from
the same topic are never treated as independent observations.

The pilot compares Direct and Winner with two seeds and may only check row
completeness, hard-decoding validity, and exact formatting. It cannot rewrite a
prompt or change the analysis. The formal study compares Direct, Rubric, and
the frozen Winner with ten seeds.

## Outcomes

Performance outcomes are direct-human pairwise accuracy, strict three-way
accuracy, seed flip, normalised margin variance, invalidity, tokens, and
latency. Winner-minus-baseline inference uses topic-paired bootstrap confidence
intervals and permutation tests. The two Winner contrasts form a Holm family
within each scope and metric.

UQ uses the frozen first five seeds as the estimate set and the disjoint last
five as future outcomes. It tests whether sampling variance predicts future
numerical deviation, margin uncertainty predicts future human-reference error,
and pair disagreement predicts future model preference flips. The isotonic
error calibrator is fitted on the internal validation set and applied to E11
without external refitting.

E10's practical Bayesian early-stopping thresholds are also transferred to
E11. This is a pair-at-a-time serving analysis: selective accuracy must always
be reported together with coverage and mean runs. These fixed confidence
thresholds have no conformal coverage guarantee.

## Commands

```bash
python scripts/prepare_e11_webis.py
sbatch slurm/run_e11_webis.sbatch pilot
sbatch --dependency=afterok:PILOT_JOB_ID slurm/run_e11_webis.sbatch formal
sbatch --dependency=afterok:FORMAL_JOB_ID slurm/process_e11_webis.sbatch
```

The frozen source and generated-panel hashes are stored in
`configs/splits/e11_webis_manifest.json`. Formal outputs are written under
`results/uq_apo/processed/e11_webis_external/`.
