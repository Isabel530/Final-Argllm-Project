# Supplementary Experiment Lock — 4 August 2026

Locked at **2026-08-04 09:52:28 Europe/London**, before generating any new
Qwen manual-baseline or full-training scores.

## What had already been observed

- The validation winner was frozen as `p_53a4acbe525c`.
- Its Qwen internal-test pairwise accuracy (approximately `.9417`) had already
  been inspected.
- Aggregate Mistral means for Direct, Rubric, and the frozen winner had already
  been inspected.

Consequently, the additional Qwen baseline experiment is described as a
**supplementary locked-baseline comparison**, not a prospectively registered
test. The 70-claim experiment is a **post-selection full-training robustness
audit**, not a retrospective insertion into the successive-halving selection
algorithm. Neither experiment can trigger prompt rewriting or winner
reselection.

## Frozen artefacts

| Artefact | SHA-256 |
|---|---|
| Claim split | `b4edbfefadadba4947a574ece94c8dde429a98f971643fbd0e8ab4ece1c3122e` |
| Freeze manifest | `3cec33762283663beeea1d1e5478cdee74314fe4c4961cd3f91329efc7326166` |
| Winner registry | `ae7fe03c20adbef7cee02e48e06b5bc7f19b25dc0d4142babc30bd921319198a` |
| Four manual prompts | `d2ed3620535c4077afe78667927810d78a9bd98eb090ad77b83ec0ed86909a07` |
| Existing Qwen winner raw file | `34a261a45f77e588a4c53c4e64480003121d1a536411248fcabb4fc90ac02547` |
| Existing Mistral transfer raw file | `d43a80e7a8b2b1327c36f490401c580bd9f736812c449e8f3ea7e487c6f6cd0f` |

## Qwen manual controls

Direct, Role, Rubric, and Role+Rubric are evaluated on the same 50 locked
internal-test claims and ten seeds as the frozen winner, using Qwen3-8B, Hard,
No-Think, temperature `.7`, and top-p `.95`.

The four winner-minus-baseline contrasts form one Holm-corrected family within
each metric. The independent unit is the claim. We report 10,000-resample
paired claim bootstrap intervals and 100,000-resample paired sign-flip tests
for accuracy, seed flip, normalised score variance, tokens, and latency.

## Full-training audit

The two promoted candidates are evaluated on all 70 APO-training claims with
five seeds. This audit checks mini-batch overfitting and stability across the
complete optimisation split. It cannot affect the already frozen winner.

## External E9 selection rule

The primary external dataset must contain independent, human-labelled
arguments not derived from the dissertation's 149 claims; retain a recoverable
topic/claim and stance or pairwise preference; have a stable, research-usable
source; and not have been used by APO.

Format-pilot claims will be excluded from the formal external test. The frozen
winner and all four manual prompts will be tested with ten seeds. Primary UQ
transfer uses sampling variance for future numerical deviation and margin
uncertainty for future reference-order error without external recalibration;
external cross-fitted recalibration is exploratory.

The machine-readable protocol is `configs/supplementary_lock_20260804.yaml`.
