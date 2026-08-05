"""Conformance tests for sarva_foundry.train.reasoning (spec §3.6a:
reasoning/thinking-token training). The reward functions are pure
string logic, so the bar here is exhaustive edge-case coverage of what
counts as "well-formed" -- a reward function that's wrong at the edges
would silently reinforce the wrong behavior during real GRPO training,
which is a much harder bug to catch after the fact than a failing
unit test now."""

from __future__ import annotations

from sarva_foundry.train.reasoning import answer_reward, format_reward, reasoning_reward


def test_format_reward_accepts_a_well_formed_completion():
    assert format_reward("<think>2+3=5</think>5") == 1.0


def test_format_reward_rejects_missing_think_tags_entirely():
    assert format_reward("the answer is 5") == 0.0


def test_format_reward_rejects_an_unclosed_think_tag():
    assert format_reward("<think>2+3=5 the answer is 5") == 0.0


def test_format_reward_rejects_an_empty_reasoning_segment():
    assert format_reward("<think></think>5") == 0.0
    assert format_reward("<think>   </think>5") == 0.0


def test_format_reward_rejects_an_empty_answer_segment():
    assert format_reward("<think>2+3=5</think>") == 0.0
    assert format_reward("<think>2+3=5</think>   ") == 0.0


def test_format_reward_rejects_text_before_the_think_tag():
    # The format requires <think> to be the very start of the
    # completion, not just present somewhere in it.
    assert format_reward("well, <think>2+3=5</think>5") == 0.0


def test_format_reward_accepts_multiline_reasoning():
    completion = "<think>\nstep 1: 2+3\nstep 2: =5\n</think>\nthe answer is 5"
    assert format_reward(completion) == 1.0


def test_format_reward_rejects_a_repeated_closing_tag():
    # Real reward-hacking exploit caught empirically while building
    # examples/17_reasoning_token_training.py, not a hypothetical
    # concern: a first-occurrence-only version of this check let GRPO
    # training discover that padding the completion with extra
    # "</think>" copies inflated reward without answering correctly.
    completion = "<think>2</think>5</think>5</think>5</think>3+2</think>"
    assert format_reward(completion) == 0.0


def test_format_reward_rejects_a_repeated_opening_tag():
    completion = "<think>2+3=5</think>5<think>extra</think>"
    assert format_reward(completion) == 0.0


def test_answer_reward_matches_when_the_expected_answer_appears_after_think():
    completion = "<think>2+3=5</think>the answer is 5"
    assert answer_reward(completion, "5") == 1.0


def test_answer_reward_does_not_match_a_wrong_answer():
    completion = "<think>2+3=5</think>the answer is 6"
    assert answer_reward(completion, "5") == 0.0


def test_answer_reward_ignores_the_expected_answer_appearing_only_inside_think():
    # The answer must appear in the ANSWER segment, not just somewhere
    # in the reasoning -- a model that reasons about "5" but then
    # answers "6" must not be rewarded as if it got it right.
    completion = "<think>let me consider 5 as a candidate</think>the answer is 6"
    assert answer_reward(completion, "5") == 0.0


def test_answer_reward_is_zero_without_a_closing_think_tag():
    # No </think> at all means there's no defined "answer segment" --
    # this must return 0.0, not raise, since GRPO calls this on
    # arbitrary real model output during rollout, some of which will be
    # malformed by construction (that's the whole point of scoring it).
    assert answer_reward("no think tags here, just 5", "5") == 0.0


def test_answer_reward_rejects_a_repeated_closing_tag_even_if_the_answer_appears():
    # The second half of the same real reward-hacking exploit
    # format_reward's own regression test names: even after format_
    # reward was tightened to reject multiple closing tags, GRPO still
    # found that repeating "</think>" padded the text answer_reward
    # scans with extra copies of the target digit -- abandoning format
    # reward (worth less) to inflate the larger-weighted answer reward.
    # Closing it here, not just in format_reward, is what actually stops
    # the exploit (reasoning_reward sums both).
    completion = "<think>2</think>5</think>5</think>5</think>3+2</think>"
    assert answer_reward(completion, "5") == 0.0


def test_answer_reward_does_not_reward_a_wrong_answer_containing_the_right_digit():
    # A third real reward-hacking exploit, found the same way
    # sarva.eval.harness.contains_match's identical bug was: a raw
    # substring check rewards a genuinely WRONG answer whenever the
    # right digit happens to appear inside a longer wrong number.
    # Single-digit addition (this module's own example task) makes this
    # a real, not hypothetical, risk -- roughly half of all real sums
    # are two-digit (10-18). Confirmed directly before this fix:
    # answer_reward("<think>...</think>The answer is 17", "7") returned
    # 1.0, rewarding a wrong answer as if it were correct.
    completion = "<think>let me add</think>The answer is 17"
    assert answer_reward(completion, "7") == 0.0
    # A genuinely correct single-digit answer must still be rewarded.
    correct = "<think>let me add</think>The answer is 7."
    assert answer_reward(correct, "7") == 1.0


def test_answer_reward_is_not_sign_blind():
    # A fourth real reward-hacking exploit, an independent instance of
    # the identical sign-blindness bug already found and fixed in
    # sarva.eval.harness.contains_match: `\b` treats `-` as a non-word
    # character, so a word boundary already exists between a minus sign
    # and the digits that follow it -- the old `\bexpected\b` pattern
    # matched a wrong "-45" just as readily as a correct "45". Confirmed
    # directly before this fix: answer_reward("<think>...</think>The
    # answer is -45", "45") returned 1.0 -- a model that reverses
    # subtraction operand order got full training reward for a
    # numerically wrong answer, corrupting the actual RL training
    # signal, not just a benchmark report.
    wrong = "<think>let me subtract</think>The answer is -45"
    assert answer_reward(wrong, "45") == 0.0
    correct = "<think>let me subtract</think>The answer is 45"
    assert answer_reward(correct, "45") == 1.0


def test_answer_reward_is_not_decimal_blind():
    # A fifth real reward-hacking exploit, an independent instance of
    # the identical decimal-point-adjacency bug already found and fixed
    # in sarva.eval.harness.contains_match: "." is not \w, so it never
    # blocked a match on either side of the word-boundary pattern, the
    # same shape as the sign-blindness bug above. This function's own
    # word-boundary fix was copied from contains_match before THAT
    # function's own later decimal-point fix existed, so it never
    # inherited the correction. Confirmed directly before this fix:
    # answer_reward("<think>...</think>The answer is 9.5", "9") returned
    # 1.0 -- a model whose real sampled output happened to include a
    # trailing decimal (a plausible failure mode for an undertrained
    # model, not contrived) got full training reward for a numerically
    # wrong answer, corrupting the actual RL training signal via
    # examples/17_reasoning_token_training.py's real call into
    # reasoning_reward() against real sample_completion() output.
    wrong_leading = "<think>4 plus 5</think>The answer is 9.5"
    assert answer_reward(wrong_leading, "9") == 0.0
    wrong_trailing = "<think>84 divided by 7</think>The answer is 12.5"
    assert answer_reward(wrong_trailing, "12") == 0.0
    # A bare trailing period (ordinary sentence punctuation, not a
    # decimal continuation) must still reward correctly -- only a
    # period followed by another digit is a genuine decimal.
    correct = "<think>4 plus 5</think>The answer is 9."
    assert answer_reward(correct, "9") == 1.0


def test_answer_reward_does_not_deny_reward_for_an_integer_formatted_as_a_float():
    # A sixth real reward-hacking exploit, introduced by the fifth
    # bug's own fix and found by a much later fresh-eyes sweep,
    # independently of the identical gap found and fixed in
    # sarva.eval.harness.contains_match: the decimal-continuation
    # lookahead rejected ANY digit right after the decimal point,
    # including an all-zero continuation that's numerically the exact
    # same value, not a genuinely different one -- a false NEGATIVE
    # this time, not a false positive. Confirmed directly before this
    # fix: answer_reward("<think>...</think>The answer is 9.0", "9")
    # returned 0.0, even though "9.0" is mathematically exactly "9" --
    # an entirely ordinary sampled completion shape for an undertrained
    # model doing float-style arithmetic, not contrived. This silently
    # denied real training reward for a genuinely correct answer,
    # corrupting the RL signal in the opposite direction from every
    # prior bug pinned above (those all inflated reward for a wrong
    # answer; this one deflates it for a right one).
    correct_as_float = "<think>4 plus 5</think>The answer is 9.0"
    assert answer_reward(correct_as_float, "9") == 1.0
    correct_as_float_padded = "<think>84 divided by 7</think>The answer is 12.00"
    assert answer_reward(correct_as_float_padded, "12") == 1.0
    # A genuinely different decimal value must still score zero -- the
    # widened lookahead must not accidentally reward any decimal at all.
    wrong_decimal = "<think>4 plus 5</think>The answer is 9.5"
    assert answer_reward(wrong_decimal, "9") == 0.0


def test_reasoning_reward_combines_format_and_answer_with_default_weights():
    both_right = "<think>2+3=5</think>5"
    assert reasoning_reward(both_right, "5") == 1.0  # 0.3*1 + 0.7*1

    format_only = "<think>2+3=5</think>6"
    assert abs(reasoning_reward(format_only, "5") - 0.3) < 1e-9  # 0.3*1 + 0.7*0

    answer_only = "the answer is 5"  # no think tags -> format fails
    # answer_reward is also 0 here (no </think> to find an answer segment after)
    assert abs(reasoning_reward(answer_only, "5") - 0.0) < 1e-9

    neither = "the answer is 6"
    assert reasoning_reward(neither, "5") == 0.0


def test_reasoning_reward_correct_answer_always_outscores_format_only():
    # The weighting itself is the point of this test: a correctly
    # answered completion must score strictly higher than a perfectly
    # formatted but wrong one, since a wrong answer with a nice-looking
    # reasoning trace still isn't useful.
    formatted_correct = "<think>2+3=5</think>5"
    formatted_wrong = "<think>2+3=5</think>6"
    assert reasoning_reward(formatted_correct, "5") > reasoning_reward(formatted_wrong, "5")


def test_reasoning_reward_respects_custom_weights():
    completion = "<think>2+3=5</think>6"  # format-correct, answer-wrong
    reward = reasoning_reward(completion, "5", format_weight=1.0, answer_weight=0.0)
    assert reward == 1.0
