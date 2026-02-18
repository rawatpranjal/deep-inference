# Writing Guidelines: Practitioner's Guide to Deep Learning for Individual Heterogeneity

These guidelines synthesize patterns from three exemplary practitioner's guides---Nevo (2000), Cameron & Miller (2015), and Beine et al. (2016)---into actionable writing rules for revising the FLM guide. Every rule includes the principle, a bad/good example pair, and the section(s) it most urgently applies to.

---

## 1. Voice & Pronouns

**Rule:** Use "I" for opinionated advice and authority claims. Use "we" for shared work with the reader.

| Pronoun | Use for | Example |
|---------|---------|---------|
| I | Recommendations, opinions, warnings | "I believe ridge Lambda should always be the default." |
| we | Walking through derivations or code | "As we saw in Section 2, the Hessian depends on theta." |
| you | Addressing the reader's choices | "If you're working with count data, use a Poisson family." |

**Bad:** "One might consider using patience=50 for multinomial models."
**Good:** "I recommend patience=50 for any multinomial model. In my experience, patience=10 with a three-way split triggers early stopping before the network has converged."

**Priority sections:** Introduction (delete the meta-commentary on line ~42 about pronoun choice), Conclusion (convert feature catalogs to "you" address).

---

## 2. Prose Over Lists

**Rule:** Default to flowing prose. Reserve bulleted/numbered lists for genuinely enumerable items (a short list of alternatives, a set of regime conditions). Never use a list when a paragraph would work.

**Bad (Introduction, ~lines 27-36):**
> **Running Example: H&M Fashion Demand.**
> - Consumer embeddings from two-tower contrastive learning
> - Product attributes via PCA
> - Multinomial logit choice model
> - IF-corrected standard errors

**Good:**
> We build the entire guide around a single application: estimating heterogeneous price sensitivity from H&M fashion transaction data. Consumer preferences are represented as 64-dimensional embeddings learned via two-tower contrastive learning. Products enter through PCA-reduced style features and prices. A multinomial logit maps these inputs to purchase probabilities, and the influence function correction delivers valid standard errors on the average price coefficient.

**Priority sections:** Introduction (Running Example list), Estimation (Algorithm Steps 0--4), Conclusion (Other Structural Models, Other Targets).

---

## 3. Example First, Abstraction Second

**Rule:** Start every concept with the concrete H&M example, then generalize. Nevo introduces the cereal market before the BLP demand system. Our guide should introduce H&M before the structural loss.

**Bad (Framework opening):**
> "Consider a structural econometric model with loss function $\ell(y, t, \theta(x))$..."

**Good:**
> "A consumer walks into H&M's website and sees 83 dresses. She buys one. From that single purchase, what can we learn about her price sensitivity? The structural model formalizes this question. The loss function $\ell(y, t, \theta(x))$---in our case the negative log-likelihood of a multinomial logit---maps her choice $y$, the attributes of available items $t$, and her latent preferences $\theta(x)$ into a single number measuring fit."

**Priority sections:** Framework 2.1 (Setup), Framework 2.3 (Why Naive Inference Fails).

---

## 4. Earn Every Equation

**Rule:** Before any equation, state (a) the problem it solves, (b) the intuition for why this particular form works, and (c) what each symbol means in the running example. After the equation, give a one-sentence "in other words" translation.

**Bad:**
> The influence function is $\psi_i = H_i - H_{\theta,i}' \Lambda^{-1} \ell_{\theta,i}$.

**Good:**
> The naive estimate of $E[\beta_{\text{price}}]$ is just the sample average of $\hat{\beta}_{\text{price}}(X_i)$. But the neural network's regularization biases each $\hat{\beta}$ toward zero, so the average inherits that bias. The influence function corrects for this:
> $$\psi_i = H_i - H_{\theta,i}' \Lambda^{-1} \ell_{\theta,i}.$$
> In words: start with the target $H_i = \hat{\beta}_{\text{price}}(X_i)$, then subtract a correction proportional to how wrong the model's first-order conditions are at observation $i$. The matrix $\Lambda$ (the expected Hessian) translates score residuals into parameter-space corrections.

---

## 5. Consequence-First Warnings

**Rule:** Lead with the failure mode and its cost, not the abstract rule. This is Nevo's "I believe..." pattern and Cameron & Miller's "A common mistake is..." pattern.

**Bad:** "The patience parameter should not be set too low."

**Good:** "Setting patience=10 with a multinomial model is the single most common way to get 0% coverage. The three-way split reduces training data to 60%, making validation loss noisy. The network stops at epoch 15-20---far too early. I wasted two weeks debugging what turned out to be a one-line fix: patience=50."

**Priority sections:** Estimation 3.7 (Pitfalls), Estimation 3.4 (Lambda Methods).

---

## 6. Algorithm as Narrative

**Rule:** Present algorithmic steps as a story, not a numbered procedure. The reader should understand *why* each step follows the previous one.

**Bad (Estimation 3.2):**
> **Step 0:** Prepare data (encode T, define X).
> **Step 1:** K-fold cross-fitting...
> **Step 2:** Estimate Lambda...

**Good:**
> The algorithm has a natural flow. You start by encoding the data: pack alternative attributes into $T$, extract consumer embeddings into $X$, set $Y$ to the chosen alternative index. Then comes the heavy lifting---cross-fitting. The neural network trains on $K-1$ folds and predicts $\hat{\theta}(X_i)$ on the held-out fold, repeating for all $K$ folds. At this point you have $\hat{\theta}$ for every observation, but these estimates are biased. The next step corrects for that: estimate the expected Hessian $\Lambda(X)$ using ridge regression on the computed Hessians. Finally, assemble the influence function $\psi_i$ from the scores, Hessian, and target Jacobian, and compute the standard error from $\text{Var}(\psi)/n$.

---

## 7. Tables: Show, Don't Summarize

**Rule:** Every table must contain real numbers. Placeholder "[to be filled]" entries destroy reader trust. If results aren't ready, omit the table entirely.

**Application Section 4.4 (critical):** The main results table (Table 2) currently has six rows of "[to be filled]". This is the most urgent fix. Run `03_inference.py --all-params` and populate every cell.

**Simulation Table (Table 1):** Similarly awaiting real numbers from `06_simulation_study.py`.

---

## 8. The MLP Cautionary Tale

**Rule:** The most effective teaching in the current draft is the MLP Lambda story (Section 3.4): high correlation (0.997) with the oracle yet 67% coverage. Expand this pattern. Every section should have one "surprising failure" that motivates the recommended approach.

Candidates:
- **Section 2.3:** Show a real simulation where naive coverage is 5% while IF achieves 95%
- **Section 3.5:** Show training curves where patience=10 stops at epoch 17 vs patience=50 converges at epoch 180
- **Section 3.6:** Show a run where correction_ratio=2 (binary) vs 80 (multinomial)---the reader's first instinct is "that's wrong" but it's correct

---

## 9. Transition Sentences

**Rule:** End each subsection with a sentence that motivates the next one. Nevo consistently does this: "The question, of course, is how to estimate these parameters. This is what I turn to next."

**Bad:** (Section 2.2 ending) "[equation]"

**Good:** "The target $\mu^*$ is well-defined. The challenge is computing a valid standard error for $\hat{\mu}$---and that is precisely where naive approaches fail."

---

## 10. Appendix Strategy

**Rule:** The main text should be conceptual (what and why). The appendix should be computational (how). A reader skipping the appendix should understand the full argument; a reader needing code should find everything in the appendix.

**Current state:** The appendix has extensive code listings but limited commentary. Add connecting prose explaining *why* the code is structured that way (e.g., why `V[0]` instead of `V[int(y)]` for vmap compatibility).

---

## Section-by-Section Revision Priorities

### Introduction
1. Convert the "Running Example" bullet list to a flowing paragraph
2. Delete the meta-commentary about first-person voice
3. End with a clearer roadmap that previews results ("We find that...")

### Framework
1. Open Section 2.1 with the H&M purchase decision, then abstract
2. In Section 2.3, lead with a simulation figure showing 5% naive coverage
3. In Section 2.4, give economic intuition for each IF term

### Estimation
1. Rewrite Algorithm (3.2) as narrative prose
2. Expand the MLP cautionary tale in Lambda Methods (3.4)
3. Add patience=10 failure curves to Network Architecture (3.5)
4. Convert Diagnostics (3.6) from reference table to interpretive guide

### Application
1. **Fill all placeholder tables with real numbers** (most urgent)
2. Update data description to match actual sample (4,783 consumers, 83 items, 10,681 occasions)
3. Add a "what we expected vs what we found" discussion to Results
4. Expand counterfactuals with concrete dollar amounts

### Conclusion
1. Convert "Other Models" and "Other Targets" from feature catalogs to narrative
2. Replace "Three Regimes Revisited" (redundant) with forward-looking guidance
3. End with practical "next steps" for the reader

---

## Style Reference Patterns

### From Nevo (2000)
- "Probably the most straightforward way to deal with..."
- "Just to be sure, suppose that..."
- "Intuitively, this means that..."
- "In other words, [equation translation]"
- "I believe that [opinionated recommendation]"
- "Care has to be taken when..."

### From Cameron & Miller (2015)
- "A common mistake is to..."
- "Table X presents results for [concrete scenario]"
- "The key result is that..."
- "In practice, [rule of thumb]"

### From Beine et al. (2016)
- "The basic idea is..."
- "To illustrate this point, consider..."
- "It is important to note that..."
- Opening with a policy-relevant question, then showing how the model answers it

---

## Checklist for Every Section

Before considering a section complete, verify:

- [ ] Opens with H&M example or connects to it within 3 sentences
- [ ] No bullet list longer than 4 items (convert to prose)
- [ ] Every equation has a preceding motivation and following "in other words"
- [ ] At least one "surprising failure" or cautionary tale
- [ ] Ends with a transition to the next section
- [ ] Uses "I" for advice, "we" for shared work, "you" for reader address
- [ ] No placeholder values or "[to be filled]" entries
- [ ] Code examples are minimal in main text (full versions in appendix)
