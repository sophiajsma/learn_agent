"""Unit-test helpers for assignment TODOs.

These tests use fake samplers / canned responses so students can validate
logic without making LLM API calls.
"""

from xag329a_a2.methods import get_verifier

_GREEN = "\033[92m"  # bright green
_RED = "\033[91m"
_RESET = "\033[0m"


def _print_pass(message: str) -> None:
    print(f"{_GREEN}{message}{_RESET}")


def _print_fail(message: str) -> None:
    print(f"{_RED}{message}{_RESET}")


def test_evaluate_zero_shot(func):
    """Check zero-shot evaluation on fixed responses (fake sampler)."""
    verifier_for_test = get_verifier("aime25")
    examples = [
        # Correct answer
        (
            [[r"Final Answer: \boxed{42}"]],
            [{"problem": "What is 40 + 2?", "answer": "42"}],
            [1],
        ),
        # Incorrect answer
        (
            [[r"Final Answer: \boxed{1}"]],
            [{"problem": "What is 40 + 2?", "answer": "42"}],
            [0],
        ),
        # Mixed: first correct, second incorrect
        (
            [
                [r"Final Answer: \boxed{42}"],
                [r"Final Answer: \boxed{7}"],
            ],
            [
                {"problem": "What is 40 + 2?", "answer": "42"},
                {"problem": "What is 3 + 4?", "answer": "99"},
            ],
            [1, 0],
        ),
        # Multiple problems: correct, incorrect, correct
        (
            [
                [r"Final Answer: \boxed{10}"],
                [r"Final Answer: \boxed{0}"],
                [r"Final Answer: \boxed{15}"],
            ],
            [
                {"problem": "What is 5 + 5?", "answer": "10"},
                {"problem": "What is 2 + 2?", "answer": "5"},
                {"problem": "What is 7 + 8?", "answer": "15"},
            ],
            [1, 0, 1],
        ),
    ]
    errcount = 0
    for fake_responses, problems_for_test, expected in examples:

        def fake_method(prompts, _responses=fake_responses):
            assert len(prompts) == len(_responses)
            return _responses

        correctness, responses = func(
            fake_method, problems_for_test, verifier_for_test
        )
        if list(map(int, correctness)) != expected:
            errcount += 1
            _print_fail(
                f"Error for `{func.__name__}`: "
                f"expected {expected!r}, got {list(map(int, correctness))!r} "
                f"for problems={problems_for_test!r}."
            )
        if len(responses) != len(problems_for_test):
            errcount += 1
            _print_fail(
                f"Error for `{func.__name__}`: "
                f"expected {len(problems_for_test)} response groups, "
                f"got {len(responses)}."
            )
    if errcount == 0:
        _print_pass(f"No errors detected for `{func.__name__}`")
    else:
        _print_fail(f"{errcount} error(s) detected for `{func.__name__}`")


def test_get_majority_answer(func):
    """Check majority voting on fixed response lists."""
    examples = [
        # Clear majority
        (
            [
                r"Final Answer: \boxed{42}",
                r"Final Answer: \boxed{41}",
                r"Final Answer: \boxed{42}",
                r"Final Answer: \boxed{42}",
            ],
            "42",
        ),
        # Different majority
        (
            [
                r"Final Answer: \boxed{7}",
                r"Final Answer: \boxed{8}",
                r"Final Answer: \boxed{7}",
                r"Final Answer: \boxed{9}",
            ],
            "7",
        ),
        # Empty / unparseable -> empty string
        (
            ["", "no answer here"],
            "",
        ),
    ]
    errcount = 0
    for responses, expected in examples:
        predicted = func(responses)
        if predicted != expected:
            errcount += 1
            _print_fail(
                f"Error for `{func.__name__}`: "
                f"expected {expected!r}, got {predicted!r} "
                f"for responses={responses!r}."
            )
    if errcount == 0:
        _print_pass(f"No errors detected for `{func.__name__}`")
    else:
        _print_fail(f"{errcount} error(s) detected for `{func.__name__}`")


def test_llm_voting_call(lv):
    """Check that __call__ samples, builds aggregation prompts, then aggregates."""
    prompts = ["problem 1", "problem 2"]
    fake_responses = [
        ["solution A for p1", "solution B for p1"],
        ["solution A for p2", "solution B for p2"],
    ]
    fake_agg_output = [["best for p1"], ["best for p2"]]
    recorded = {}

    def fake_sampler(p):
        recorded["sampler_prompts"] = p
        return fake_responses

    def fake_aggregator(aggr_prompts):
        recorded["aggr_prompts"] = aggr_prompts
        return fake_agg_output

    lv.sampler = fake_sampler
    lv.aggregator = fake_aggregator

    errcount = 0
    result = lv(prompts)
    expected_aggr = lv._create_aggregation_prompts(prompts, fake_responses)

    if recorded.get("sampler_prompts") != prompts:
        errcount += 1
        _print_fail(
            f"Error for `LLMVoting.__call__`: "
            f"expected sampler prompts {prompts!r}, "
            f"got {recorded.get('sampler_prompts')!r}."
        )
    if recorded.get("aggr_prompts") != expected_aggr:
        errcount += 1
        _print_fail(
            f"Error for `LLMVoting.__call__`: "
            f"aggregator prompts did not match "
            f"`_create_aggregation_prompts` output."
        )
    if result != fake_agg_output:
        errcount += 1
        _print_fail(
            f"Error for `LLMVoting.__call__`: "
            f"expected return {fake_agg_output!r}, got {result!r}."
        )
    if errcount == 0:
        _print_pass("No errors detected for `LLMVoting.__call__`")
    else:
        _print_fail(f"{errcount} error(s) detected for `LLMVoting.__call__`")


def test_self_improvement_system(sis):
    """Check all TODO methods: generate, critique, regenerate, improve, evaluate."""
    errcount = 0

    # --- _generate_initial_solution ---
    recorded = {}

    def fake_generator(problem):
        recorded["gen"] = problem
        return [["INITIAL"]]

    sis.generator = fake_generator
    out = sis._generate_initial_solution("P1")
    if out != "INITIAL" or recorded.get("gen") != "P1":
        errcount += 1
        _print_fail(
            f"Error for `_generate_initial_solution`: "
            f"expected return 'INITIAL' and generator called with 'P1', "
            f"got return {out!r}, recorded={recorded!r}."
        )

    # --- _critique_solution ---
    recorded = {}

    def fake_critic(prompt):
        recorded["crit"] = prompt
        return [["FEEDBACK"]]

    sis.critic = fake_critic
    out = sis._critique_solution("P1", "SOL1")
    if (
        out != "FEEDBACK"
        or "P1" not in recorded.get("crit", "")
        or "SOL1" not in recorded.get("crit", "")
    ):
        errcount += 1
        _print_fail(
            f"Error for `_critique_solution`: "
            f"expected return 'FEEDBACK' and prompt containing 'P1' and 'SOL1', "
            f"got return {out!r}, prompt={recorded.get('crit')!r}."
        )

    # --- _regenerate_solution ---
    recorded = {}

    def fake_regenerator(prompt):
        recorded["regen"] = prompt
        return [["IMPROVED"]]

    sis.regenerator = fake_regenerator
    out = sis._regenerate_solution("P1", "SOL1", "FB1")
    if out != "IMPROVED" or any(
        x not in recorded.get("regen", "") for x in ("P1", "SOL1", "FB1")
    ):
        errcount += 1
        _print_fail(
            f"Error for `_regenerate_solution`: "
            f"expected return 'IMPROVED' and prompt containing 'P1', 'SOL1', 'FB1', "
            f"got return {out!r}, prompt={recorded.get('regen')!r}."
        )

    # --- improve_solution (end-to-end with mocked LLM callables) ---
    call_order = []

    def fake_generator2(p):
        call_order.append("gen")
        return [["ORIG"]]

    def fake_critic2(p):
        call_order.append("crit")
        return [["FB"]]

    def fake_regenerator2(p):
        call_order.append("regen")
        return [["NEW"]]

    sis.generator = fake_generator2
    sis.critic = fake_critic2
    sis.regenerator = fake_regenerator2
    results = sis.improve_solution("P1")
    if call_order != ["gen", "crit", "regen"]:
        errcount += 1
        _print_fail(
            f"Error for `improve_solution`: "
            f"expected call order ['gen', 'crit', 'regen'], got {call_order!r}."
        )
    expected_results = {
        "problem": "P1",
        "original_solution": "ORIG",
        "feedback": "FB",
        "improved_solution": "NEW",
    }
    if results != expected_results:
        errcount += 1
        _print_fail(
            f"Error for `improve_solution`: "
            f"expected {expected_results!r}, got {results!r}."
        )

    # --- evaluate_improvement ---
    cases = [
        # wrong -> correct => improvement True
        (
            {
                "original_solution": r"Final Answer: \boxed{1}",
                "improved_solution": r"Final Answer: \boxed{42}",
            },
            "42",
            True,
        ),
        # correct -> correct => False
        (
            {
                "original_solution": r"Final Answer: \boxed{42}",
                "improved_solution": r"Final Answer: \boxed{42}",
            },
            "42",
            False,
        ),
        # wrong -> wrong => False
        (
            {
                "original_solution": r"Final Answer: \boxed{1}",
                "improved_solution": r"Final Answer: \boxed{2}",
            },
            "42",
            False,
        ),
    ]
    for results_dict, answer, expected_imp in cases:
        ev = sis.evaluate_improvement(results_dict, answer)
        if (
            "original_correct" not in ev
            or "improved_correct" not in ev
            or "improvement" not in ev
        ):
            errcount += 1
            _print_fail(
                f"Error for `evaluate_improvement`: "
                f"missing keys in return value {ev!r}."
            )
            continue
        if bool(ev["improvement"]) != expected_imp:
            errcount += 1
            _print_fail(
                f"Error for `evaluate_improvement`: "
                f"for answer={answer!r}, results={results_dict!r}: "
                f"expected improvement={expected_imp!r}, got {ev!r}."
            )

    if errcount == 0:
        _print_pass("No errors detected for `SelfImprovementSystem`")
    else:
        _print_fail(f"{errcount} error(s) detected for `SelfImprovementSystem`")
