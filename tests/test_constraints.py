from argstrength.constraints import make_prefix_allowed_tokens_fn


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


def test_constraint_uses_complete_output_trie_and_allows_shared_prefixes():
    tokenizer = CharacterTokenizer()
    callback = make_prefix_allowed_tokens_fn(
        tokenizer=tokenizer,
        prompt_length=3,
        allowed_outputs=["Score: 1", "Score: 10"],
    )
    prompt = [7, 8, 9]
    assert callback(0, prompt) == [ord("S")]

    score_one = tokenizer.encode("Score: 1", add_special_tokens=False)
    allowed = callback(0, prompt + score_one)
    assert tokenizer.eos_token_id in allowed
    assert ord("0") in allowed

    score_ten = tokenizer.encode("Score: 10", add_special_tokens=False)
    assert callback(0, prompt + score_ten) == [tokenizer.eos_token_id]


def test_constraint_stops_invalid_path():
    callback = make_prefix_allowed_tokens_fn(
        tokenizer=CharacterTokenizer(),
        prompt_length=1,
        allowed_outputs=["Score: 0"],
    )
    assert callback(0, [99, ord("X")]) == [0]
