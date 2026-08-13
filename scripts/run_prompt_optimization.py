#!/usr/bin/env python3
"""Run the closed-loop, uncertainty-aware discrete prompt search."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from argstrength.apo import (  # noqa: E402
    PromptCandidate,
    deduplicate_texts,
    representative_failures,
    select_survivors,
)
from argstrength.apo_experiment import (  # noqa: E402
    read_candidate_registry,
    run_candidate_evaluation,
    summarise_candidate_evaluations,
    write_candidate_registry,
)
from argstrength.data import load_main_dataset  # noqa: E402
from argstrength.embedding import (  # noqa: E402
    HashingTextEmbedder,
    TransformerTextEmbedder,
)
from argstrength.experimental_design import (  # noqa: E402
    load_claim_splits,
    screening_batch,
)
from argstrength.optimizer_model import HuggingFacePromptOptimizer  # noqa: E402
from argstrength.runner import load_yaml  # noqa: E402
from argstrength.tabular import write_csv, write_json  # noqa: E402


def _make_embedder(config):
    if config.get("backend") == "hashing":
        return HashingTextEmbedder(dimensions=int(config.get("dimensions", 256)))
    if config.get("backend") != "transformer":
        raise ValueError("embedding.backend must be transformer or hashing")
    return TransformerTextEmbedder(
        config["model_name"],
        revision=config.get("revision"),
        cache_dir=config.get("cache_dir"),
        device=config.get("device"),
        batch_size=int(config.get("batch_size", 32)),
        max_length=int(config.get("max_length", 512)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uq_apo.yaml")
    parser.add_argument(
        "--initial-candidates",
        type=Path,
        default=ROOT / "results/uq_apo/candidates/iteration_0.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/uq_apo/search"
    )
    parser.add_argument("--rounds", type=int, help="Override config.search.rounds")
    parser.add_argument(
        "--hashing-embeddings",
        action="store_true",
        help="Dry-run fallback only; dissertation experiments should use transformer embeddings.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    search = config["search"]
    rounds = args.rounds or int(search["rounds"])
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    candidates = read_candidate_registry(args.initial_candidates)
    dataset_path = ROOT / config["dataset"]
    items = load_main_dataset(dataset_path)
    split_document = load_claim_splits(ROOT / config["locked_split"], items)
    train_ids = split_document["splits"]["apo_train"]
    screen = config["stages"]["search_screening"]
    generation = config["generation"]
    models = load_yaml(ROOT / "configs/models.yaml")["models"]
    optimiser_key = config["optimizer_model"]
    optimiser_config = dict(models[optimiser_key])
    optimiser_name = optimiser_config.pop("name")
    optimiser_config.pop("thinking_modes", None)

    embedding_config = dict(search["embedding"])
    if args.hashing_embeddings:
        embedding_config = {"backend": "hashing", "dimensions": 256}
    embedder = _make_embedder(embedding_config)
    history = []
    try:
        for iteration in range(rounds):
            round_dir = args.output_dir / f"round_{iteration}"
            raw_path = round_dir / "screening_raw.jsonl"
            claim_ids = screening_batch(
                items,
                train_ids,
                iteration=iteration,
                batch_size=int(screen["claims"]),
                seed=int(config["split_seed"]),
            )
            run_candidate_evaluation(
                models_path=ROOT / "configs/models.yaml",
                dataset_path=dataset_path,
                model_key=config["task_model"],
                candidates=candidates,
                claim_ids=claim_ids,
                seeds=screen["seeds"],
                output_path=raw_path,
                stage="search_screening",
                decoding=generation["decoding"],
                thinking=bool(generation["thinking"]),
                do_sample=bool(generation["do_sample"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                max_new_tokens=int(generation["max_new_tokens"]),
            )
            from argstrength.apo_experiment import read_jsonl

            evaluations, failures, pair_details = summarise_candidate_evaluations(
                read_jsonl(raw_path)
            )
            survivor_evaluations = select_survivors(
                evaluations, keep=int(search["survivors_after_screening"])
            )
            by_id = {candidate.prompt_id: candidate for candidate in candidates}
            survivors = [by_id[row.prompt_id] for row in survivor_evaluations]

            summary_rows = []
            for evaluation in evaluations:
                row = evaluation.to_dict()
                row.pop("per_claim_accuracy", None)
                row["selected"] = evaluation.prompt_id in {
                    item.prompt_id for item in survivor_evaluations
                }
                summary_rows.append(row)
            write_csv(round_dir / "candidate_summary.csv", summary_rows)
            write_csv(round_dir / "pair_diagnostics.csv", pair_details)

            cases_by_parent = {
                parent.prompt_id: representative_failures(
                    failures.get(parent.prompt_id, ()),
                    embedder,
                    max_cases=int(search["failure_representatives"]),
                    similarity_threshold=float(
                        search["failure_similarity_threshold"]
                    ),
                )
                for parent in survivors
            }
            optimiser = HuggingFacePromptOptimizer(
                optimiser_name, **optimiser_config
            )
            proposals: list[PromptCandidate] = []
            rejected = []
            try:
                optimiser_generation = search["optimizer_generation"]
                for parent_index, parent in enumerate(survivors):
                    try:
                        rewrites = optimiser.rewrite(
                            parent,
                            cases_by_parent[parent.prompt_id],
                            proposal_count=int(search["proposals_per_parent"]),
                            seed=int(config["split_seed"]) + 1009 * iteration + parent_index,
                            max_new_tokens=int(
                                optimiser_generation["proposal_max_new_tokens"]
                            ),
                            temperature=float(optimiser_generation["temperature"]),
                            top_p=float(optimiser_generation["top_p"]),
                        )
                    except ValueError as exc:
                        rejected.append(
                            {
                                "parent_id": parent.prompt_id,
                                "instruction": parent.instruction,
                                "reason": str(exc),
                                "optimizer_raw_outputs": list(
                                    getattr(exc, "raw_outputs", ())
                                ),
                            }
                        )
                        continue
                    for rewrite in rewrites:
                        try:
                            proposals.append(
                                PromptCandidate(
                                    instruction=rewrite.instruction,
                                    iteration=iteration + 1,
                                    source="uq_guided_llm_rewrite",
                                    parent_ids=(parent.prompt_id,),
                                    metadata={
                                        "critique": rewrite.critique,
                                        "source_failure_ids": list(
                                            rewrite.source_failure_ids
                                        ),
                                    },
                                )
                            )
                        except (TypeError, ValueError) as exc:
                            rejected.append(
                                {
                                    "parent_id": parent.prompt_id,
                                    "instruction": rewrite.instruction,
                                    "reason": str(exc),
                                }
                            )
            finally:
                optimiser.close()

            combined = survivors + proposals
            kept = deduplicate_texts(
                [candidate.instruction for candidate in combined],
                embedder,
                similarity_threshold=float(search["candidate_similarity_threshold"]),
            )
            candidates = [combined[index] for index in kept]
            registry = args.output_dir / "candidates" / f"iteration_{iteration + 1}.jsonl"
            write_candidate_registry(registry, candidates)
            write_json(round_dir / "rejected_proposals.json", rejected)
            round_record = {
                "iteration": iteration,
                "screening_claim_ids": claim_ids,
                "screening_seeds": screen["seeds"],
                "evaluated_prompt_ids": [row.prompt_id for row in evaluations],
                "survivor_prompt_ids": [row.prompt_id for row in survivors],
                "proposal_prompt_ids": [row.prompt_id for row in proposals],
                "next_prompt_ids": [row.prompt_id for row in candidates],
                "candidate_registry": str(registry),
            }
            write_json(round_dir / "round_manifest.json", round_record)
            history.append(round_record)
    finally:
        embedder.close()

    final_registry = args.output_dir / "candidates_for_promotion.jsonl"
    write_candidate_registry(final_registry, candidates)
    write_json(
        args.output_dir / "search_history.json",
        {
            "rounds": history,
            "note": (
                "Final proposals are not winners yet. Evaluate all of them in the "
                "promotion and validation stages before selecting one prompt."
            ),
        },
    )
    print(f"Candidates requiring promotion: {final_registry}")


if __name__ == "__main__":
    main()
