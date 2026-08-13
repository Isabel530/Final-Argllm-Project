# E10: Bayesian-Conformal Uncertainty-Aware Adaptive Sampling

## Status and research question

E10 is a **post-freeze exploratory experiment** designed after the E5--E9
results had been inspected. It cannot change the frozen APO winner and must not
be presented as a prospectively registered or confirmatory test.

The research question is:

> Can repeated LLM-judge sampling be allocated adaptively, so that easy
> argument pairs stop early while uncertain pairs receive more samples or an
> explicit abstention, without hiding the resulting coverage--risk trade-off?

## Bayesian preference model

For every argument pair and sampling checkpoint, each score difference is
mapped to one of three reference-neutral outcomes:

1. the first argument wins;
2. the scores tie;
3. the second argument wins.

The analysis implementation stores pairs in reference-oriented order for later
accuracy calculation, but stopping uses only the largest posterior category
probability and is invariant to swapping the two argument labels. Reference
correctness is evaluated only after the policy decides.

A symmetric Dirichlet prior with concentration `.5` is combined with the
observed outcome counts. The posterior mean supplies three smoothed preference
probabilities. This is a task-specific Bayesian UQ estimator; it is not a new
general Bayesian theorem.

## Claim-grouped split-conformal calibration

The 29 APO-validation claims are the calibration set. Each claim contributes
one nonconformity score: the largest `1 - P(true label)` over its six argument
pairs and all checkpoints (`k = 3, 5, 7, 10`). A finite-sample split-conformal
quantile is fitted for nominal coverage levels `.95`, `.90`, and `.80`.

Using the maximum makes calibration simultaneous over the inspected pairs and
checkpoints and keeps the claim, rather than the correlated pair, as the
calibration unit. Coverage relies on exchangeability between calibration and
test claims. E9 intentionally probes domain shift, so any external coverage
failure is a result rather than something to conceal.

Conformal label-set coverage is not the same as conditional error control among
only answered pairs. The analysis therefore reports both prediction-set
coverage and empirical selective accuracy.

## Adaptive policy

The policy observes seeds in the frozen order and checks after 3, 5, 7, and 10
runs. A whole claim stops early only when all six pairwise conformal prediction
sets are singletons. At the final checkpoint, singleton pairs are answered and
non-singleton pairs abstain.

The policy is compared with fixed 3-, 5-, and 10-run baselines using:

- pair coverage and abstention;
- selective reference and pairwise accuracy;
- marginal pair-set and simultaneous claim coverage;
- complete-order rate;
- mean runs and score generations per claim;
- generation saving relative to ten fixed runs.

## Existing-data evaluation

- Calibration: Qwen frozen winner on 29 validation claims.
- Exploratory internal evaluation: Qwen frozen winner on 50 internal claims.
- Exploratory shift evaluation: Qwen frozen winner on E9 IBM broad and
  near-tie panels, without external recalibration.

Because the internal and E9 results were already viewed before E10 was
designed, these analyses are exploratory. A confirmatory claim requires a new,
untouched external argument-quality dataset and a frozen threshold.

## Command and outputs

```bash
python scripts/analyse_adaptive_uq.py
```

Outputs are written to `results/uq_apo/processed/adaptive_sampling/`:

- `conformal_thresholds.csv`;
- `calibration_claim_scores.csv`;
- `adaptive_pair_decisions.csv`;
- `adaptive_claim_decisions.csv`;
- `adaptive_summary.csv`;
- `fixed_budget_summary.csv`;
- `analysis_manifest.json`.
