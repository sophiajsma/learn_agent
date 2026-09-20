#!/usr/bin/env python3
"""Gradescope / local grader for XAG329A Assignment 2."""

import os
import sys
import unittest

from graderUtil import graded, CourseTestRunner, GradedTestCase
from autograde_utils import (
    load_notebook_module,
    json_from_tagged_cell,
)

### BEGIN_HIDE ###
### END_HIDE ###

SUBMISSION_NOTEBOOK = os.path.join(
    os.path.dirname(__file__), "submission", "assignment2.ipynb"
)
SOLUTION_NOTEBOOK = os.path.join(
    os.path.dirname(__file__), "solution", "assignment2.ipynb"
)

# Instantiating student classes constructs LiteLLM clients; tests replace callables
# with fakes and never hit the network. Allow construction without a real key.
os.environ.setdefault("OPENROUTER_API_KEY", "gradescope-dummy-key")

# Ensure submission package is importable (xag329a_a2).
_SUBMISSION_DIR = os.path.join(os.path.dirname(__file__), "submission")
if _SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, _SUBMISSION_DIR)


def _load_assignment(path, module_name):
    return load_notebook_module(path, module_name=module_name)


#########
# TESTS #
#########


class Test_1(GradedTestCase):
    """Zero-shot evaluation (10 points)."""

    def setUp(self):
        self.nb = _load_assignment(SUBMISSION_NOTEBOOK, "assignment2_submission")
        self.evaluate_zero_shot = self.nb.evaluate_zero_shot
        self.sol_nb = None
        if os.path.isfile(SOLUTION_NOTEBOOK):
            try:
                self.sol_nb = _load_assignment(SOLUTION_NOTEBOOK, "assignment2_solution")
            except Exception:
                self.sol_nb = None

    @graded(timeout=30)
    def test_0(self):
        """1-0-basic: evaluate_zero_shot correctness on fixed fake responses"""
        from xag329a_a2.methods import get_verifier

        verifier = get_verifier("aime25")
        examples = [
            (
                [[r"Final Answer: \boxed{42}"]],
                [{"problem": "What is 40 + 2?", "answer": "42"}],
                [1],
            ),
            (
                [[r"Final Answer: \boxed{1}"]],
                [{"problem": "What is 40 + 2?", "answer": "42"}],
                [0],
            ),
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
        for fake_responses, problems, expected in examples:

            def fake_method(prompts, _responses=fake_responses):
                self.assertEqual(len(prompts), len(_responses))
                return _responses

            correctness, responses = self.evaluate_zero_shot(
                fake_method, problems, verifier
            )
            self.assertEqual(list(map(int, correctness)), expected)
            self.assertEqual(len(responses), len(problems))

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_2(GradedTestCase):
    """Majority voting (30 points)."""

    def setUp(self):
        self.nb = _load_assignment(SUBMISSION_NOTEBOOK, "assignment2_submission")
        self.MajorityVoting = self.nb.MajorityVoting
        # Instantiation builds LiteLLM clients but does not call the API.
        self.mv = self.MajorityVoting(
            model="openrouter/dummy",
            system_prompt="test",
            n_samples=4,
            temperature=0.7,
        )

    @graded(timeout=30)
    def test_0(self):
        """2-0-basic: majority answer clear majority"""
        responses = [
            r"Final Answer: \boxed{42}",
            r"Final Answer: \boxed{41}",
            r"Final Answer: \boxed{42}",
            r"Final Answer: \boxed{42}",
        ]
        self.assertEqual(self.mv._get_majority_answer(responses), "42")

    @graded(timeout=30)
    def test_1(self):
        """2-1-basic: majority answer different majority"""
        responses = [
            r"Final Answer: \boxed{7}",
            r"Final Answer: \boxed{8}",
            r"Final Answer: \boxed{7}",
            r"Final Answer: \boxed{9}",
        ]
        self.assertEqual(self.mv._get_majority_answer(responses), "7")

    @graded(timeout=30)
    def test_2(self):
        """2-2-basic: majority answer empty / unparseable"""
        self.assertEqual(self.mv._get_majority_answer(["", "no answer here"]), "")

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_3(GradedTestCase):
    """Best-of-N / LLM voting (30 points)."""

    def setUp(self):
        self.nb = _load_assignment(SUBMISSION_NOTEBOOK, "assignment2_submission")
        self.LLMVoting = self.nb.LLMVoting
        self.lv = self.LLMVoting(
            model="openrouter/dummy",
            system_prompt="test",
            n_samples=2,
            temperature=0.7,
        )

    @graded(timeout=30)
    def test_0(self):
        """3-0-basic: LLMVoting.__call__ samples then aggregates"""
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

        self.lv.sampler = fake_sampler
        self.lv.aggregator = fake_aggregator

        result = self.lv(prompts)
        expected_aggr = self.lv._create_aggregation_prompts(prompts, fake_responses)
        self.assertEqual(recorded.get("sampler_prompts"), prompts)
        self.assertEqual(recorded.get("aggr_prompts"), expected_aggr)
        self.assertEqual(result, fake_agg_output)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_4(GradedTestCase):
    """Self-improvement with feedback (45 points)."""

    def setUp(self):
        self.nb = _load_assignment(SUBMISSION_NOTEBOOK, "assignment2_submission")
        self.SelfImprovementSystem = self.nb.SelfImprovementSystem
        self.sis = self.SelfImprovementSystem(
            model="openrouter/dummy",
            system_prompt="test",
            temperature=0.7,
        )

    @graded(timeout=30)
    def test_0(self):
        """4-0-basic: _generate_initial_solution uses generator"""
        recorded = {}

        def fake_generator(problem):
            recorded["gen"] = problem
            return [["INITIAL"]]

        self.sis.generator = fake_generator
        out = self.sis._generate_initial_solution("P1")
        self.assertEqual(out, "INITIAL")
        self.assertEqual(recorded.get("gen"), "P1")

    @graded(timeout=30)
    def test_1(self):
        """4-1-basic: _critique_solution includes problem and solution"""
        recorded = {}

        def fake_critic(prompt):
            recorded["crit"] = prompt
            return [["FEEDBACK"]]

        self.sis.critic = fake_critic
        out = self.sis._critique_solution("P1", "SOL1")
        self.assertEqual(out, "FEEDBACK")
        self.assertIn("P1", recorded.get("crit", ""))
        self.assertIn("SOL1", recorded.get("crit", ""))

    @graded(timeout=30)
    def test_2(self):
        """4-2-basic: _regenerate_solution includes problem, solution, feedback"""
        recorded = {}

        def fake_regenerator(prompt):
            recorded["regen"] = prompt
            return [["IMPROVED"]]

        self.sis.regenerator = fake_regenerator
        out = self.sis._regenerate_solution("P1", "SOL1", "FB1")
        self.assertEqual(out, "IMPROVED")
        prompt = recorded.get("regen", "")
        self.assertIn("P1", prompt)
        self.assertIn("SOL1", prompt)
        self.assertIn("FB1", prompt)

    @graded(timeout=30)
    def test_3(self):
        """4-3-basic: improve_solution call order and return keys"""
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

        self.sis.generator = fake_generator2
        self.sis.critic = fake_critic2
        self.sis.regenerator = fake_regenerator2
        results = self.sis.improve_solution("P1")
        self.assertEqual(call_order, ["gen", "crit", "regen"])
        self.assertEqual(
            results,
            {
                "problem": "P1",
                "original_solution": "ORIG",
                "feedback": "FB",
                "improved_solution": "NEW",
            },
        )

    @graded(timeout=60)
    def test_4(self):
        """4-4-basic: evaluate_improvement wrong->correct is improvement"""
        cases = [
            (
                {
                    "original_solution": r"Final Answer: \boxed{1}",
                    "improved_solution": r"Final Answer: \boxed{42}",
                },
                "42",
                True,
            ),
            (
                {
                    "original_solution": r"Final Answer: \boxed{42}",
                    "improved_solution": r"Final Answer: \boxed{42}",
                },
                "42",
                False,
            ),
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
            ev = self.sis.evaluate_improvement(results_dict, answer)
            self.assertIn("original_correct", ev)
            self.assertIn("improved_correct", ev)
            self.assertIn("improvement", ev)
            self.assertEqual(bool(ev["improvement"]), expected_imp)

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_5(GradedTestCase):
    """Tagged JSON eval summaries (accuracy thresholds are hidden)."""

    ### BEGIN_HIDE ###
    ### END_HIDE ###


if __name__ == "__main__":
    # Student-readable local run (also used by make grade_A2 via graderUtil).
    assignment = unittest.TestSuite()
    assignment.addTest(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    )
    CourseTestRunner(gradescope=False).run(assignment)
