# Reliable LLM Argument-Strength Assessment

Reproducible experimental pipeline for the dissertation study:

> Do Large Language Models Reliably Assess Argument Strength? A Controlled
> Study of Model, Prompt, Decoding, and Argument-Representation Sensitivity

The repository separates input validation, controlled prompts, generation,
raw observations, metrics, and statistical analysis. Model weights remain in
the shared Hugging Face cache and are not committed.

## Study design

The confirmatory main study compares three model families, four prompt
conditions, and three generation seeds. It uses non-thinking,
hard-constrained decoding only after separate output-length and
hard-versus-soft validation studies. Thinking is evaluated separately because
a hard constraint begins at the first generated token and therefore cannot
also permit visible reasoning before the final score.

Prompt conditions are standardised to receive the same claim, argument,
stance, task definition, and 0--100 output scale. They form a controlled
Role × Rubric 2×2 design:

| Condition | Role | Rubric |
|---|---:|---:|
| `direct` | no | no |
| `role` | yes | no |
| `rubric` | no | yes |
| `role_rubric` | yes | yes |

The label `rubric` is deliberate: the experiment does not rerun the OPRO
optimisation algorithm. `prompt_legacy.py` preserves the inherited templates,
while `prompt.py` provides legacy-shaped access to the controlled versions.

## Important data semantics

The operative stance is always read from `is_effective_support`:

```python
stance = "support" if item["is_effective_support"] else "attack"
```

`is_support` is retained only as source metadata because it is null for part
of the dataset. Missing or malformed `is_effective_support` values stop the
run instead of silently selecting a default.

The four arguments are a synthetically designed reference ordering. Results
therefore measure agreement with that ordering, not agreement with an
objective or human gold-standard notion of argument strength.

## Repository layout

```text
configs/                 model and study definitions
datasets/                immutable study inputs
src/argstrength/         reusable experiment library
scripts/                 command-line entry points
tests/                   regression tests for known failure modes
results/raw/             one JSON object per model generation
results/processed/       reproducibly derived metrics
results/tables/          dissertation-ready tables
results/figures/         dissertation-ready figures
```

Each raw record preserves the complete output, parsed score or parsing error,
exact-format status, claim, argument, stance, model, prompt, decoding method,
thinking mode, seed, runtime, generated-token count, and max-token truncation
flag. `format_ok` means the final answer has exactly `Score: <integer>`;
`score_parse_ok` means one unambiguous labelled score can be extracted. A soft
output may therefore be extractable without being exactly formatted. There is
no fallback score for invalid output.

Runs append one record at a time. The restart key includes study, model,
prompt, decoding, thinking, seed, claim, and argument, so Think and No-Think
records cannot incorrectly skip one another.

## Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

On the Imperial GPU environment, reuse the existing model cache:

```bash
export HF_HOME=/vol/bitbucket/xz3725/huggingface_cache
export TRITON_CACHE_DIR=/vol/bitbucket/xz3725/triton_cache
```

The same paths are recorded in `configs/models.yaml`. Change the YAML if the
project is moved to another system.

## Required execution order

1. Validate both datasets without using a GPU:

   ```bash
   python scripts/validate_data.py
   ```

2. Run unit tests:

   ```bash
   pytest -q
   ```

3. Validate hard-constrained decoding against short soft decoding on the
   configured 20-claim sample, using exactly 10 support and 10 attack claims.
   The smaller output-length pilot uses 10 claims: 5 support and 5 attack.

   ```bash
   python scripts/run_study.py \
     --study decoding_validation \
     --models qwen3_8b \
     --output results/raw/decoding_validation_qwen3_8b.jsonl
   ```

   Before that validation, `output_length_pilot` provides a 10-claim soft
   pilot with a 512-token ceiling for both modes. Inspect
   `compliance.csv`, especially extraction, token count, and truncation, before
   freezing these values.

4. Run the optional reasoning study separately. It uses soft decoding because
   hard decoding would suppress visible reasoning:

   ```bash
   python scripts/run_study.py \
     --study reasoning_study \
     --models qwen3_8b \
     --output results/raw/reasoning_study_qwen3_8b.jsonl
   ```

5. After the pilots have fixed the protocol, run one main-study file per
   model. Never let concurrent jobs append to the same JSONL:

   ```bash
   python scripts/run_study.py --study main_study --models mistral_7b \
     --output results/raw/main_study_mistral_7b.jsonl
   python scripts/run_study.py --study main_study --models qwen3_8b \
     --output results/raw/main_study_qwen3_8b.jsonl
   python scripts/run_study.py --study main_study --models gpt_oss_20b \
     --output results/raw/main_study_gpt_oss_20b.jsonl
   ```

   The prepared cluster script takes `STUDY` and `MODEL_KEY` and derives the
   independent output name. These commands are examples only; no job is
   submitted automatically:

   ```bash
   sbatch --export=NIL slurm/run_study.sbatch main_study mistral_7b
   sbatch --export=NIL slurm/run_study.sbatch main_study qwen3_8b
   sbatch --export=NIL slurm/run_study.sbatch main_study gpt_oss_20b

   `--export=NIL` avoids Slurm login-environment retrieval failures.
   ```

6. Analyse all completed model files together:

   ```bash
   python scripts/analyse.py \
     results/raw/main_study_mistral_7b.jsonl \
     results/raw/main_study_qwen3_8b.jsonl \
     results/raw/main_study_gpt_oss_20b.jsonl \
     --output-dir results/processed/main_study
   ```

7. Run the smaller representation study only after confirming that its sampled
   summary/paraphrase groups operationalise the intended RQ:

   ```bash
   python scripts/run_representation_study.py
   python scripts/analyse.py results/raw/representation_study.jsonl
   ```

## Metrics and inference

The primary results average each argument's three seed scores first and then
compute claim-level metrics. Per-seed metrics are retained as diagnostics;
seeds are never treated as independent claims.

Primary descriptive metrics are:

- tie-aware claim-level Spearman correlation;
- pairwise accuracy, with a score tie receiving 0.5 credit;
- strict pairwise accuracy, requiring the preferred argument to lead by at
  least 10 score points (`--strict-margin` can change this declared threshold);
- pairwise tie rate.

An all-tied claim has undefined Spearman rather than an invented zero.
Uncertainty intervals resample claims. Prompt contrasts use paired
claim-level sign-flip permutation tests and Holm family-wise-error correction
within each study × model × decoding × thinking × stance family.

`scripts/analyse.py` writes:

| File | Purpose |
|---|---|
| `compliance.csv` | exact format, extraction, tokens, truncation, runtime |
| `claim_metrics_per_seed.csv` | diagnostic claim metrics for every seed |
| `claim_metrics_seed_average.csv` | primary claim metrics after seed averaging |
| `aggregate_metrics.csv` | overall primary means and claim-bootstrap CIs |
| `aggregate_by_stance.csv` | separate support/attack primary results |
| `repeat_stability.csv` | claim-level score variation, pair consistency, flips |
| `repeat_stability_aggregate.csv` | stability summary by experimental cell |
| `prompt_comparisons.csv` | paired tests, interaction, CIs, Holm-adjusted p |
| `representation_sensitivity.csv` | summary/expanded-form sensitivity |

## Configuration notes

- `configs/models.yaml` controls model IDs, cache, dtype, and quantisation.
- `configs/experiments.yaml` controls datasets, prompts, decoding, seeds,
  sample sizes, and generation parameters.
- Mistral is declared No-Think only. Qwen3 and GPT-OSS support both modes.
- `output_length_pilot`, `decoding_validation`, `main_study`, and
  `reasoning_study` are separate so pilot decisions cannot silently alter the
  confirmatory estimand.
- Main generation is non-thinking and hard constrained; reasoning generation
  is soft constrained with mode-specific token limits.
- The representation study begins with 50 groups and the `rubric` prompt to
  control GPU cost. This is editable after design validation.

## Legacy material

`prompt_legacy.py`, `chatgpt_opro_ablation_llm_manager_legacy.py`, and
`README_gitlab_template.md` are preserved for provenance. They are not formal
study entry points and should not be used to generate new dissertation results.
