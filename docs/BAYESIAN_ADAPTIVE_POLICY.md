# Practical Bayesian Adaptive Policy

This companion to E10 applies fixed Dirichlet-posterior confidence thresholds
of `.60`, `.70`, `.80`, and `.90`. A claim is checked after 3, 5, 7, and 10
runs. It stops when all six pairs have a unique outcome above the threshold;
at the last checkpoint, unresolved pairs abstain.

These policies expose an empirical accuracy--coverage--cost frontier. They do
**not** carry the simultaneous split-conformal coverage guarantee. The strict
claim-grouped conformal policy and the practical Bayesian policy are therefore
reported in separate tables.

Run:

```bash
python scripts/analyse_bayesian_adaptive.py
```

The outputs are `bayesian_adaptive_summary.csv`, pair/claim decision tables,
and `bayesian_policy_manifest.json` under
`results/uq_apo/processed/adaptive_sampling/`.

