from argstrength.webis_analysis import (
    build_pair_metrics,
    evaluate_pairwise_bayesian_adaptive,
    gold_label,
)


def _raw(prompt, seed, argument_id, score):
    return {
        "prompt_condition": prompt,
        "claim_id": 0,
        "argument_id": argument_id,
        "seed": seed,
        "score": score,
        "score_parse_ok": True,
        "metadata": {"prompt_id": prompt, "prompt_label": "winner"},
    }


PAIR = {
    "pair_id": "pair-1",
    "external_claim_id": 0,
    "topic_id": 3,
    "source_topic": "A claim",
    "stance": "support",
    "argument_a_id": "rank_1",
    "argument_b_id": "rank_2",
    "logical": "A",
    "rhetorical": "A",
    "dialectical": "B",
    "majority_label": "A",
    "combined_label": "A",
    "agreement_count": 2,
    "unanimous": False,
}


def test_gold_scopes_keep_primary_and_drop_nonunanimous():
    assert gold_label(PAIR, "majority_decisive") == "A"
    assert gold_label(PAIR, "unanimous_decisive") is None
    assert gold_label(PAIR, "rhetorical_decisive") == "A"


def test_direct_human_pair_accuracy_uses_a_b_orientation():
    rows = []
    for seed, left, right in [(1, 80, 20), (2, 70, 30), (3, 75, 25)]:
        rows.extend([_raw("winner", seed, "rank_1", left), _raw("winner", seed, "rank_2", right)])
    metrics = build_pair_metrics(rows, [PAIR], prompt_ids=["winner"], seeds=[1, 2, 3])
    primary = [row for row in metrics if row["scope"] == "majority_decisive"]
    assert len(primary) == 1
    assert primary[0]["pairwise_accuracy"] == 1.0
    assert primary[0]["seed_flip"] == 0.0


def test_bayesian_pair_policy_reports_coverage_and_cost():
    rows = []
    for seed in range(1, 11):
        rows.extend([_raw("winner", seed, "rank_1", 90), _raw("winner", seed, "rank_2", 10)])
    decisions, summary = evaluate_pairwise_bayesian_adaptive(
        rows,
        [PAIR],
        prompt_id_value="winner",
        seed_order=list(range(1, 11)),
        checkpoints=[3, 5, 7, 10],
        confidence_thresholds=[0.6],
        prior_concentration=0.5,
    )
    assert decisions[0]["human_correct"] is True
    assert summary[0]["pair_coverage"] == 1.0
    assert summary[0]["mean_runs_per_pair"] == 3.0
    assert summary[0]["generation_saving_vs_10_runs"] == 0.7
