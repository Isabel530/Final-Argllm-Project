from argstrength.external_analysis import (
    aggregate_topic_metrics,
    external_prompt_contrasts,
)


def _claim(prompt, topic, panel, accuracy):
    return {
        "model": "qwen3_8b",
        "prompt_id": prompt,
        "prompt_label": "winner" if prompt == "w" else "direct",
        "claim_id": hash((prompt, topic, panel)) % 100000,
        "stance": "support",
        "source_topic": topic,
        "source_group_id": topic,
        "panel": panel,
        "n_seeds": 10,
        "pairwise_accuracy": accuracy,
        "seed_flip_rate": 0.1,
        "score_variance": 0.01,
        "mean_tokens": 6.0,
        "mean_seconds": 0.2,
        "invalid_rate": 0.0,
    }


def test_external_inference_aggregates_by_topic():
    rows = []
    for topic in ("t1", "t2"):
        for panel in ("broad", "near_tie"):
            rows.append(_claim("w", topic, panel, 1.0))
            rows.append(_claim("d", topic, panel, 0.5))
    topic_rows = [
        row
        for panel in ("all", "broad", "near_tie")
        for row in aggregate_topic_metrics(rows, panel=panel)
    ]
    contrasts = external_prompt_contrasts(
        topic_rows,
        winner_id="w",
        baseline_ids=["d"],
        bootstrap_resamples=100,
        permutation_resamples=100,
        seed=7,
    )
    accuracy = [
        row
        for row in contrasts
        if row["panel"] == "broad" and row["metric"] == "pairwise_accuracy"
    ][0]
    assert accuracy["n_topics"] == 2
    assert accuracy["difference_winner_minus_baseline"] == 0.5
