# Optional Analysis: What to Expect

These supplementary descriptions are for reflection, not graded scoring. Use **your** notebook numbers as evidence. The notes below describe **qualitative expectations and reasoning**—not target accuracies—so you can check whether your interpretation is on solid conceptual ground.

**Overall expectation:** Test-time compute usually helps on hard math, but *how* you spend API calls often matters as much as *how many*. Expect tradeoffs among accuracy, cost, latency, and failure modes—not a single “best” method for every budget.

---

## Task 1: Quantitative Analysis (Optional)

1. **Accuracy Rankings**

   **What to do:** Rank zero-shot, majority voting (report the best $n$ you tried), LLM voting, and self-improvement using the accuracies from *your* run.

   **What to expect (reasoning, not a fixed order):**
   - A single-sample baseline (zero-shot / majority with $n=1$) is usually the floor.
   - Parallel sampling (majority) often improves over the baseline, but gains can saturate.
   - An aggregator that *reads* multiple solutions (LLM voting) can beat raw majority at a similar sample budget when the correct answer is present but not the plurality.
   - A short critique–revise loop (self-improvement) can land between—or occasionally above—many-sample majority, because feedback redirects a wrong first attempt instead of only averaging correlated samples.
   - Absolute ranking can shift with model, temperature, and sampling noise. Prefer explaining *why* your order makes sense over matching a canonical leaderboard.

2. **Cost-Effectiveness**

   **What to do:** Compute accuracy gain **per additional API call** for majority voting ($n=1 \to n=16$) and for self-improvement (typically 3 calls: generate → critique → regenerate). Compare which gives better returns in *your* data.

   **What to expect:**
   - Majority voting often shows **modest** gain per extra call once samples are correlated (same model, same mistakes).
   - Self-improvement often shows **stronger** gain per extra call when the critic catches fixable errors, because two follow-up calls can change the reasoning path.
   - Also check the *peak* along the majority curve (best $n$), not only $n=16$—the last few samples may add cost with little benefit.

3. **Self-Improvement Analysis**

   **What to do:** Report (a) improved vs. zero-shot (and, separately, original vs. improved *within* the RLEF cell), (b) which error types show up most in critiques, (c) what fraction of problems flipped incorrect → correct (and any regressions).

   **What to expect:**
   - Prefer the **within-pipeline** delta (original → improved in the same RLEF run) when judging the method. Comparing to a separate zero-shot cell is useful but noisy—different samples, same model.
   - On AIME-style problems, critiques often highlight **completeness** (missing steps / unfinished arguments) and **conceptual setup** mistakes more than isolated arithmetic; logical gaps and calculation errors still appear.
   - Expect a **subset** of problems to improve, not all; a healthy run often has few or no regressions (correct → incorrect), but that is not guaranteed if the critic is wrong or overconfident.

---

## Task 2: Method-Specific Insights (Optional)

1. **Majority Voting**

   **What to expect:**
   - Plot or list accuracy vs. $n$. Look for where **diminishing returns** begin: the $n$ after which accuracy flattens or wobbles instead of climbing.
   - The **maximum** need not be at the largest $n$. When errors are shared across samples, more votes mostly reinforce the same wrong answer.
   - Reasoning to articulate: independence assumption fails → majority is not a free lunch.

2. **LLM Voting**

   **What to expect:**
   - At the same generation budget (e.g. $n=16$), LLM voting often **matches or beats** majority because selection can favor a correct minority solution.
   - Cost is nearly the same as majority at that $n$, plus **one** aggregation call—so a large accuracy gap (if you see one) is an argument for smarter use of the last call, not just more samples.
   - If LLM voting does *not* help in your run, ask whether candidates were too similar, or whether the aggregator prompt failed to compare carefully.

3. **Self-Improvement**

   **What to expect:**
   - **Vs. parallel test-time compute:** majority / LLM voting spend budget in parallel; RLEF spends it sequentially (solve → critique → revise). RLEF can be competitive at low call count; large parallel budgets (especially with a strong aggregator) may still win on raw accuracy.
   - **Feedback vs. independent samples:** feedback is targeted and often efficient per call, but serial and critic-dependent. Independent samples parallelize easily but waste budget when samples are highly correlated.
   - **Error categories that guide improvement:** conceptual and completeness feedback is usually most *actionable* for regeneration (it tells the model what to change). Calculation flags help mainly when the plan was already right. Prefer specific, category-tagged critiques over “try again.”
