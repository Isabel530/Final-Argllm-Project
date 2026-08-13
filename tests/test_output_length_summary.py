from scripts.summarise_output_length import summarise


def _row(tokens, *, claim_id=1, stance="support", truncated=False, parsed=True):
    return {
        "claim_id": claim_id,
        "stance": stance,
        "prompt_condition": "direct",
        "raw_output": "x" * tokens,
        "generated_tokens": tokens,
        "hit_max_new_tokens": truncated,
        "score_parse_ok": parsed,
        "format_ok": parsed,
    }


def test_summary_counts_tokens_and_rounds_complete_maximum():
    report = summarise([
        _row(500),
        _row(700, claim_id=2, stance="attack"),
    ])
    assert report["n_claims"] == 2
    assert report["claims_by_stance"] == {"attack": 1, "support": 1}
    assert report["total_generated_tokens"] == 1200
    assert report["max_generated_tokens"] == 700
    assert report["rounded_max_power_of_two"] == 1024


def test_summary_does_not_recommend_from_censored_lengths():
    report = summarise([_row(40960, truncated=True, parsed=False)])
    assert report["truncation_rate"] == 1.0
    assert report["rounded_max_power_of_two"] is None
