# Flashcard Drill Set — Statistical Inference + ML Fundamentals

**How to use:** Read the **Q**, say the answer *out loud* (dictate it), then check against **A**.
The ⚠️ line is the trap *you personally hit* in the diagnostic — those are the ones to over-learn.
Goal: recite the A cleanly and automatically, in your own words. Re-drill until zero hesitation.

---

## Cluster A — Statistical Inference (TOP PRIORITY)

### A1. What is a p-value?
**A:** The probability of observing a result **at least as extreme** as what you saw, **assuming the null hypothesis is true.** It's `P(data | null true)`.
⚠️ It is NOT a confidence level, NOT "95% sure we can reject," and NOT `P(null is true | data)`. Never flip the conditional.

### A2. p-value vs. significance level (α)
**A:** **α** is the threshold you set *in advance* — the false-positive rate you'll tolerate *if the null were true* (e.g. 0.05). The **p-value** is computed *from the data* you actually got. You reject the null when p < α.
⚠️ p = 0.03 does NOT mean "3% chance this is a false positive." That conflates the p-value with α.

### A3. Type I vs. Type II error
**A:** **Type I (α)** = false positive — rejecting a true null (seeing an effect that isn't real). **Type II (β)** = false negative — failing to reject a false null (missing a real effect).
Mnemonic: Type I = crying wolf; Type II = missing the wolf.

### A4. What is statistical power?
**A:** Power = **1 − β** = the probability of **detecting a real effect** when one truly exists. Higher power → less likely to miss real effects. Driven up by: larger n, larger true effect, lower variance, higher α.
⚠️ Power is a *design input* you choose (usually 80%), not something you read off after the test.

### A5. Confidence interval — what does a 95% CI mean?
**A:** The range of **plausible values for the true parameter** consistent with the data. Precisely: if you repeated the experiment many times and built a CI each time, ~95% of those intervals would contain the true value.
⚠️ NOT "95% of data points fall in this range." NOT (strictly) "95% probability the true value is in *this* interval" — that's a Bayesian credible interval.

### A6. CI and hypothesis testing — the shortcut
**A:** A 95% CI that **excludes 0** ⇔ statistically significant at α = 0.05. If the CI **includes 0**, you can't rule out "no effect" (or even harm) → not significant.
Applied: CI of [−0.5%, +4.5%] crosses 0 → not significant → lean no-ship / gather more data (it's underpowered).

### A7. Central Limit Theorem — what does it say?
**A:** The distribution of the **sample mean** approaches a **normal distribution** as n grows, **regardless of the shape of the underlying data** (given finite variance).
⚠️ CLT does NOT assume the underlying data is normal — it *manufactures* normality in the mean from non-normal data. That's its whole power.

### A8. CLT — why can I A/B test a skewed metric (e.g. revenue/user)?
**A:** Because you're doing inference on the **mean**, not on individual values. Even if revenue/user is heavily skewed, the sampling distribution of the *mean* is ~normal for large n, so z/t-tests, SEs, and CIs are valid.
Nuance that scores: heavy skew needs a **larger n**, and outliers inflate variance → often **winsorize/cap** or **bootstrap**.

### A9. Statistical vs. practical significance
**A:** Statistical significance = the effect is unlikely to be noise. Practical significance = the effect is **big enough to matter** for the business. With huge n, a trivial effect can be statistically significant but useless.
Always separate the two — mention it unprompted.

### A10. The "peeking" problem
**A:** Checking results repeatedly and stopping the first time p < 0.05 **inflates the Type I error rate** — each look is another chance to cross the line by luck (true FP rate can hit 20–30%+). Under the null, peek forever → *guaranteed* eventual false "significance."
Fix: predefine a fixed sample size / horizon, evaluate once. If you must monitor live → **sequential testing / alpha-spending / always-valid p-values**.
⚠️ The reason is repeated-looks inflating α — NOT just "the effect might not persist."

---

## Cluster B — Bayes / Probability

### B1. Bayes via natural frequencies (the recipe)
**A:** Don't juggle formulas — **imagine a concrete population and count.**
1. Pick a big round population (e.g. 1,000,000).
2. Apply the **base rate** → how many have the condition.
3. Apply the **true-positive rate** → sick who test positive.
4. Apply the **false-positive rate** → healthy who test positive.
5. `P(sick | positive) = true positives ÷ all positives.`
Example: 1-in-10k disease, 99% accurate test → 99 / (99 + 9,999) ≈ **1%.**

### B2. Base-rate neglect
**A:** When a condition is **rare**, a positive test is dominated by **false positives** — the small error rate applied to the huge negative population produces more false positives than there are true positives. So "99% accurate" ≠ "99% likely sick."
Through-line: this is exactly why **precision is low on rare-event detection** (scam/fraud classifiers).

---

## Cluster C — Experimentation Stats

### C1. What drives A/B sample size? (4 inputs + directions)
**A:** (1) **Variance** ↑ → n ↑. (2) **MDE** (min detectable effect) ↓ → n ↑. (3) **α** ↓ (stricter) → n ↑. (4) **Power** ↑ → n ↑.
Key intuition: `n ∝ variance / MDE²` — **MDE is squared**, so halving the effect you want to detect **quadruples** the sample.
Duration: run ≥ one full business cycle (usually 1–2 weeks) even if n is hit sooner (weekly seasonality, novelty).

### C2. Multiple comparisons
**A:** Testing many metrics at α = 0.05 inflates the chance of a false positive. With 10 independent tests: `1 − 0.95^10 ≈ 40%` chance of ≥1 false positive.
Fix: designate **one primary metric** up front; for the rest apply a correction — **Bonferroni** (α ÷ #tests, conservative) or **Benjamini-Hochberg / FDR** (controls false discovery rate, less conservative).

### C3. Novelty vs. primacy effect
**A:** **Novelty** = users engage more just because something is *new* (early lift fades). **Primacy** = existing users react to *any* change (up or down) before settling. Both distort early A/B readings.
Diagnose by splitting **new vs. returning users**; evaluate the **steady-state** effect after a full cycle.

### C4. Network effects / interference
**A:** When one user's treatment affects control users (social, marketplace, messaging), **user-level randomization is contaminated** and biases the estimate.
Fix: **cluster/geo randomization** (randomize groups — cities, social clusters — instead of individuals).

---

## Cluster D — ML Fundamentals

### D1. Bias-variance tradeoff
**A:** **Bias** = error from a model too *simple* to capture the real pattern → **underfitting.** **Variance** = error from a model too *sensitive* to the training data → **overfitting.** Total error ≈ bias² + variance + irreducible noise. More complexity lowers bias but raises variance; you minimize the *sum*.
⚠️ Name the terms explicitly: overfitting = high variance, underfitting = high bias.

### D2. Fixing an overfit (high-variance) model
**A:** (1) **More training data** (most reliable). (2) **Regularization** (L1/L2). (3) **Reduce complexity** / feature selection / lower depth. (4) **Cross-validation** to tune honestly. (5) **Early stopping**, **dropout** (NNs), **bagging/ensembles**.
**L1 vs L2:** L1 (Lasso) → sparsity / feature selection; L2 (Ridge) → shrinks weights smoothly.

### D3. Precision vs. recall
**A:** **Precision** = of what we *flagged* positive, what fraction *actually is* → `TP / (TP + FP)` ("how often we cry wolf"). **Recall** = of all *actual* positives, what fraction we *caught* → `TP / (TP + FN)` ("how much we miss").
Which to favor = depends on the **cost of each error** and the action taken.

### D4. Why accuracy fails on imbalanced data
**A:** With rare positives (e.g. 0.1% scams), a model that predicts "negative" for *everything* scores ~99.9% accuracy while catching **zero** positives — the majority-class baseline is trivially high.
Use instead: **precision/recall, F-beta, and PR-AUC** (precision-recall AUC is the right summary for rare-positive problems — better than ROC-AUC).

### D5. Logistic vs. linear regression for classification
**A:** Use **logistic regression** for binary classification — it outputs a **probability in [0,1]** (via the sigmoid), which you threshold to 0/1. **Linear regression** outputs unbounded values and is for continuous targets.
⚠️ "Linear regression to output a probability" is wrong — that's logistic regression.

### D6. The 6-step modeling framing template ("build a model to do X")
**A:** **Frame → Data/Labels → Features → Model → Evaluate → Deploy/Monitor.**
1. **Frame:** classification vs regression, the unit, **define the label precisely**, and the **cost of each error** (drives the metric).
2. **Data/Labels:** sources, expert + weak labels (reports/enforcement), handle **class imbalance**.
3. **Features:** general + **domain-specific** signals (e.g. similarity-to-known-brand for impersonation).
4. **Model:** simple baseline (logistic) → stronger (gradient-boosted trees); note interpretability needs.
5. **Evaluate:** right metric for imbalance (**PR-AUC, precision/recall**), proper **train/val/test split**, watch **leakage**.
6. **Deploy/Monitor:** how it's used (auto-action vs human-review queue), threshold by cost asymmetry, monitor **drift + adversarial adaptation**.
⚠️ Lead with framing (don't jump to data/tools), and always close with deployment/monitoring.
