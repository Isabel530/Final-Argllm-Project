"""Uncertainty-aware automatic prompt optimisation (UQ-APO).

This is embedding-assisted *discrete* optimisation: every candidate remains a
readable instruction block.  Embeddings cluster failure cases and remove
near-duplicate candidates; they are not continuous "soft prompt" parameters.
The optimisation loop is closed by injectable evaluation and rewrite
callbacks, allowing local Hugging Face or API-based optimiser models without
changing the search algorithm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import statistics
from typing import Any, Callable, Iterable, Sequence

from .embedding import (
    TextEmbedder,
    cluster_embeddings,
    cluster_representatives,
    deduplicate_texts,
)
from .prompts import ALLOWED_SCORE_OUTPUTS, SCORE_PREFIX, PromptSpec
from .schemas import PromptCondition, Stance


RELATION_MARKER = "<<RELATION_RULE>>"

_TASK = """Evaluate the strength of the argument in performing its stated role with respect to the claim.

Claim: {claim}
Role: The argument is intended to {relation} the claim.
Argument: {argument}"""

_SCALE = """Assign an integer score from 0 to 100, where:
0 means the argument provides no valid or effective reason for its stated role;
100 means the argument provides an exceptionally strong and effective reason for its stated role.

Respond with exactly:
Score: <integer>"""

_ROLE = (
    "You are an impartial expert analyst in argumentation. Assess arguments "
    "consistently and do not favour supporting or attacking arguments."
)

_RUBRIC = """Assess the argument using the following criteria:
1. Factuality: Are its factual statements accurate and credible?
2. Relevance: Does it directly address the claim?
3. Logical quality: Does its reasoning justify its intended conclusion?
4. Effectiveness: How strongly does it perform its stated supporting or attacking role?

<<RELATION_RULE>>

Give one holistic score. Do not output separate criterion scores."""


def _clean_instruction(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("instruction must be a string")
    cleaned = value.strip()
    if "Claim:" in cleaned or "Argument:" in cleaned or "Score: <integer>" in cleaned:
        raise ValueError(
            "candidate instruction must not rewrite the frozen task, input fields, or scale"
        )
    return cleaned


def candidate_id(instruction: str) -> str:
    payload = f"uq-apo-v1\n{_clean_instruction(instruction)}".encode("utf-8")
    return f"p_{hashlib.sha256(payload).hexdigest()[:12]}"


@dataclass(frozen=True)
class PromptCandidate:
    instruction: str
    iteration: int
    source: str
    parent_ids: tuple[str, ...] = ()
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.iteration) is not int or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be non-empty")
        cleaned = _clean_instruction(self.instruction)
        object.__setattr__(self, "instruction", cleaned)
        object.__setattr__(self, "prompt_id", candidate_id(cleaned))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initial_prompt_candidates() -> list[PromptCandidate]:
    """Seed APO with the same interpretable 2x2 Role × Rubric family."""

    return [
        PromptCandidate("", 0, "controlled_seed", label="direct"),
        PromptCandidate(_ROLE, 0, "controlled_seed", label="role"),
        PromptCandidate(_RUBRIC, 0, "controlled_seed", label="rubric"),
        PromptCandidate(
            f"{_ROLE}\n\n{_RUBRIC}",
            0,
            "controlled_seed",
            label="role_rubric",
        ),
    ]


def build_candidate_prompt(
    candidate: PromptCandidate,
    *,
    claim: str,
    argument: str,
    stance: str | Stance,
) -> PromptSpec:
    """Render a candidate while freezing task input, scale, and output syntax."""

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be non-empty")
    if not isinstance(argument, str) or not argument.strip():
        raise ValueError("argument must be non-empty")
    stance = stance if isinstance(stance, Stance) else Stance(stance)
    relation = "support" if stance is Stance.SUPPORT else "attack"
    relation_rule = (
        "For a supporting argument, a strong argument should provide valid "
        "reasons that the claim is true or credible."
        if stance is Stance.SUPPORT
        else
        "For an attacking argument, a strong argument should provide valid "
        "reasons that the claim is false, doubtful, or less credible."
    )
    instruction = candidate.instruction.replace(RELATION_MARKER, relation_rule)
    task = _TASK.format(
        claim=claim.strip(), relation=relation, argument=argument.strip()
    )
    if candidate.label in {"role", "role_rubric"} and instruction.startswith(_ROLE):
        guidance = instruction[len(_ROLE) :].strip()
        blocks = [_ROLE, task]
        if guidance:
            blocks.append(guidance)
    else:
        blocks = [task]
        if instruction:
            blocks.append(instruction)
    blocks.append(_SCALE)
    # The enum value is not used as the APO identity. Raw records store the
    # hash-based candidate ID in prompt_condition and metadata.prompt_id.
    return PromptSpec(
        condition=PromptCondition.RUBRIC,
        text="\n\n".join(blocks),
        output_prefix=SCORE_PREFIX,
        allowed_outputs=ALLOWED_SCORE_OUTPUTS,
    )


@dataclass(frozen=True)
class CandidateEvaluation:
    prompt_id: str
    stage: str
    n_claims: int
    n_seeds: int
    pairwise_accuracy: float
    accuracy_se: float
    score_variance: float
    seed_flip_rate: float
    brier: float | None
    mean_tokens: float
    mean_seconds: float
    invalid_rate: float = 0.0
    per_claim_accuracy: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("pairwise_accuracy", "seed_flip_rate", "invalid_rate"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.n_claims <= 0 or self.n_seeds < 2:
            raise ValueError("evaluation needs claims and at least two seeds")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["per_claim_accuracy"] = list(self.per_claim_accuracy)
        return result


@dataclass(frozen=True)
class FailureCase:
    prompt_id: str
    claim_id: int
    claim: str
    stance: str
    better_argument: str
    worse_argument: str
    rank_distance: int
    pairwise_accuracy: float
    uncertainty: float
    seed_flip_rate: float
    seed_margins: tuple[float, ...] = ()

    @property
    def priority(self) -> float:
        adjacent_bonus = 0.25 if self.rank_distance == 1 else 0.0
        return (
            1.5 * (1.0 - self.pairwise_accuracy)
            + self.uncertainty
            + self.seed_flip_rate
            + adjacent_bonus
        )

    def embedding_text(self) -> str:
        return (
            f"Stance: {self.stance}. Claim: {self.claim} Better reference argument: "
            f"{self.better_argument} Worse reference argument: {self.worse_argument} "
            f"Outcome accuracy={self.pairwise_accuracy:.3f}; "
            f"uncertainty={self.uncertainty:.3f}; flip={self.seed_flip_rate:.3f}."
        )


@dataclass(frozen=True)
class RewriteProposal:
    instruction: str
    critique: str
    source_failure_ids: tuple[str, ...] = ()


def _objective_vector(evaluation: CandidateEvaluation) -> tuple[float, ...]:
    return (
        evaluation.pairwise_accuracy,
        -evaluation.score_variance,
        -evaluation.seed_flip_rate,
        -(evaluation.brier if evaluation.brier is not None else 1.0),
        -evaluation.mean_tokens,
        -evaluation.mean_seconds,
        -evaluation.invalid_rate,
    )


def dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    left_values = _objective_vector(left)
    right_values = _objective_vector(right)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_front(evaluations: Sequence[CandidateEvaluation]) -> list[CandidateEvaluation]:
    """Return candidates not dominated on accuracy, stability, UQ, and cost."""

    return [
        candidate
        for candidate in evaluations
        if not any(
            other.prompt_id != candidate.prompt_id and dominates(other, candidate)
            for other in evaluations
        )
    ]


def conservative_utility(
    evaluation: CandidateEvaluation,
    *,
    confidence_z: float = 1.96,
    flip_weight: float = 0.15,
    variance_weight: float = 0.10,
    brier_weight: float = 0.10,
    invalid_weight: float = 0.25,
    token_budget: float = 50.0,
    cost_weight: float = 0.02,
) -> float:
    """Accuracy lower confidence bound penalised by instability and cost."""

    accuracy_lcb = evaluation.pairwise_accuracy - confidence_z * evaluation.accuracy_se
    calibration = evaluation.brier if evaluation.brier is not None else 0.25
    token_penalty = max(0.0, evaluation.mean_tokens - token_budget) / max(token_budget, 1)
    return (
        accuracy_lcb
        - flip_weight * evaluation.seed_flip_rate
        - variance_weight * evaluation.score_variance
        - brier_weight * calibration
        - invalid_weight * evaluation.invalid_rate
        - cost_weight * token_penalty
    )


def select_survivors(
    evaluations: Sequence[CandidateEvaluation],
    *,
    keep: int,
) -> list[CandidateEvaluation]:
    """Successive-halving selection preserving Pareto candidates where possible."""

    if keep <= 0:
        raise ValueError("keep must be positive")
    by_id = {evaluation.prompt_id: evaluation for evaluation in evaluations}
    if len(by_id) != len(evaluations):
        raise ValueError("candidate evaluations must have unique prompt IDs")
    pareto = sorted(pareto_front(evaluations), key=conservative_utility, reverse=True)
    ordered = pareto + sorted(
        (row for row in evaluations if row.prompt_id not in {item.prompt_id for item in pareto}),
        key=conservative_utility,
        reverse=True,
    )
    return ordered[: min(keep, len(ordered))]


def representative_failures(
    failures: Sequence[FailureCase],
    embedder: TextEmbedder,
    *,
    max_cases: int = 8,
    similarity_threshold: float = 0.72,
) -> list[FailureCase]:
    """Select high-priority medoids from different semantic failure clusters."""

    if max_cases <= 0 or not failures:
        return []
    ordered = sorted(failures, key=lambda row: row.priority, reverse=True)
    # Bound quadratic clustering cost while retaining the most useful failures.
    pool = ordered[: max(max_cases * 5, max_cases)]
    vectors = embedder.encode([row.embedding_text() for row in pool])
    clusters = cluster_embeddings(vectors, similarity_threshold=similarity_threshold)
    medoids = cluster_representatives(vectors, clusters)
    representatives = [pool[index] for index in medoids]
    return sorted(representatives, key=lambda row: row.priority, reverse=True)[:max_cases]


RewriteFunction = Callable[
    [PromptCandidate, Sequence[FailureCase]], Sequence[RewriteProposal]
]
EvaluationFunction = Callable[
    [Sequence[PromptCandidate], int], tuple[Sequence[CandidateEvaluation], dict[str, Sequence[FailureCase]]]
]


@dataclass
class SearchRound:
    iteration: int
    candidates: list[PromptCandidate]
    evaluations: list[CandidateEvaluation]
    survivors: list[str]
    proposed: list[PromptCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "survivors": self.survivors,
            "proposed": [candidate.to_dict() for candidate in self.proposed],
        }


class UncertaintyAwarePromptOptimizer:
    """Closed-loop, checkpoint-friendly UQ-guided discrete search."""

    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        evaluate: EvaluationFunction,
        rewrite: RewriteFunction,
        population_size: int = 4,
        proposals_per_parent: int = 3,
        failure_representatives: int = 8,
        failure_similarity_threshold: float = 0.72,
        candidate_similarity_threshold: float = 0.92,
    ) -> None:
        if population_size <= 0 or proposals_per_parent <= 0:
            raise ValueError("population_size and proposals_per_parent must be positive")
        self.embedder = embedder
        self.evaluate = evaluate
        self.rewrite = rewrite
        self.population_size = population_size
        self.proposals_per_parent = proposals_per_parent
        self.failure_representatives = failure_representatives
        self.failure_similarity_threshold = failure_similarity_threshold
        self.candidate_similarity_threshold = candidate_similarity_threshold

    def run(
        self,
        initial: Sequence[PromptCandidate],
        *,
        rounds: int,
    ) -> tuple[list[PromptCandidate], list[SearchRound]]:
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        population = list(initial)
        history: list[SearchRound] = []
        for iteration in range(rounds):
            evaluations, failures_by_prompt = self.evaluate(population, iteration)
            evaluations = list(evaluations)
            if {row.prompt_id for row in evaluations} != {
                row.prompt_id for row in population
            }:
                raise ValueError("evaluator did not return exactly one result per candidate")
            survivor_evaluations = select_survivors(
                evaluations, keep=self.population_size
            )
            by_id = {candidate.prompt_id: candidate for candidate in population}
            survivors = [by_id[row.prompt_id] for row in survivor_evaluations]

            proposals: list[PromptCandidate] = []
            for parent in survivors:
                representatives = representative_failures(
                    failures_by_prompt.get(parent.prompt_id, ()),
                    self.embedder,
                    max_cases=self.failure_representatives,
                    similarity_threshold=self.failure_similarity_threshold,
                )
                rewritten = list(self.rewrite(parent, representatives))[
                    : self.proposals_per_parent
                ]
                proposals.extend(
                    PromptCandidate(
                        proposal.instruction,
                        iteration=iteration + 1,
                        source="uq_guided_rewrite",
                        parent_ids=(parent.prompt_id,),
                        metadata={
                            "critique": proposal.critique,
                            "source_failure_ids": list(proposal.source_failure_ids),
                        },
                    )
                    for proposal in rewritten
                )

            combined = survivors + proposals
            kept_indices = deduplicate_texts(
                [candidate.instruction for candidate in combined],
                self.embedder,
                similarity_threshold=self.candidate_similarity_threshold,
            )
            population = [combined[index] for index in kept_indices]
            # The next screening stage can contain more candidates than the
            # target population; successive halving trims them after evidence.
            history.append(
                SearchRound(
                    iteration=iteration,
                    candidates=list(by_id.values()),
                    evaluations=evaluations,
                    survivors=[candidate.prompt_id for candidate in survivors],
                    proposed=proposals,
                )
            )
        return population, history


def accuracy_standard_error(per_claim_accuracy: Sequence[float]) -> float:
    if len(per_claim_accuracy) < 2:
        return 0.0
    return statistics.stdev(per_claim_accuracy) / math.sqrt(len(per_claim_accuracy))
