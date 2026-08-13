from argstrength.parsing import parse_score


def test_strict_parser_accepts_only_exact_labelled_integer():
    assert parse_score("Score: 73").score == 73
    assert parse_score(" score: 0 \n").score == 0
    assert not parse_score("The answer is Score: 73").ok
    assert not parse_score("73").ok


def test_parser_never_invents_default_score():
    result = parse_score("No valid answer")
    assert not result.ok
    assert result.score is None


def test_out_of_range_and_multiple_scores_fail():
    assert not parse_score("Score: 101").ok
    assert not parse_score("Score: 10\nScore: 20", strict=False).ok


def test_soft_parser_accepts_one_labelled_score_after_reasoning():
    result = parse_score("Short reasoning.\nFinal score: 61/100", strict=False)
    assert result.ok
    assert result.score == 61


def test_gpt_oss_final_channel_is_parsed_as_the_final_answer():
    raw = "Reasoning mentions Score: 20.assistantfinalScore: 95"
    strict = parse_score(raw, strict=True)
    soft = parse_score(raw, strict=False)
    assert strict.ok and strict.score == 95
    assert soft.ok and soft.score == 95


def test_gpt_oss_final_channel_keeps_range_and_format_validation():
    assert not parse_score("Reasoning.assistantfinalScore: 101").ok
    assert not parse_score("Reasoning.assistantfinalThe answer is 70").ok
