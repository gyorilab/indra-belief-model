# Belief scoring: methods

A self-contained account of how a belief score is computed, what each step
assumes, and what we propose to change. Every term is defined at first use.
Numbers are measured from this repository unless marked otherwise.

---

## 1. The quantity being estimated

INDRA represents a text-mined biological assertion as a **Statement** — a typed
relation such as `RPS6KA1 [Phosphorylation] YBX1 @S102` — which owns a list of
**Evidence** objects. Each Evidence carries one source sentence and the identity
of the reader or database that produced it (`source_api`, e.g. `reach`,
`signor`).

Define the binary outcome

$$Y = \begin{cases} 1 & \text{the Statement is a correct extraction} \\ 0 & \text{otherwise}\end{cases}$$

"Correct" means a human curator, reading the source sentence, would judge that
the sentence asserts the claimed relation between the claimed entities. It is a
property of the (claim, sentence) pair, not of the underlying biology.

The **belief score** is an estimate of $\Pr(Y = 1)$ given everything read. Two
grains are involved:

| grain | unit | question |
|---|---|---|
| per-evidence | one (Statement, Evidence) pair | does *this sentence* support *this claim*? |
| per-statement | one Statement | is the Statement correct, pooling all its evidence? |

The remainder is organised as three layers: producing a per-evidence number
(§2), making that number mean what it says (§3), and pooling across evidence
(§4). §5 gives a reduction result linking the proposal to the deployed system,
§6 turns source reliability into an empirical question, §7 covers evaluation,
and §8 lists the assumptions.

---

## 2. Layer 1 — the per-evidence score

### 2.1 What is deployed now

A language model is shown the claim, entity aliases, and the evidence sentence,
and is asked to emit two categorical labels:

- **verdict** $\in$ {`correct`, `incorrect`}
- **confidence** $\in$ {`high`, `medium`, `low`}

The pair is mapped to a number through a fixed six-cell lookup table
(`scorers/_shared.py`):

| | high | medium | low |
|---|---|---|---|
| correct | 0.95 | 0.80 | 0.65 |
| incorrect | 0.05 | 0.20 | 0.35 |

Two properties of this design matter.

**It is a verbalized confidence measure.** In the uncertainty-quantification
literature, methods that obtain uncertainty by *asking the model to state its own
confidence* are called **reflexive** or **verbalized** methods, as distinct from
**information-based** methods, which read the model's output probability
distribution. Vashurin et al. (TACL 2025) benchmark both families and report
that reflexive methods "in general do not demonstrate good performance", and
that the reflexive baseline P(True) "in most cases is not better than random"
for models below frontier scale.

**The six values are assigned, not estimated.** No derivation for 0.95, 0.80,
0.65, 0.35, 0.20 or 0.05 exists anywhere in the repository. They are an *ordinal*
encoding of two labels, and treating them as probabilities is a modelling choice
made silently.

We observe both failure modes directly. Under the verdict-only prompt, three of
four models emit `high` on 98–99% of executions — a confidence field that takes
one value carries no information. And when the same eight executions are scored
through a model that exposes its output distribution, the model's own posterior
ranges from 0.679 to 0.99997 across rows that the table maps identically to 0.95.

### 2.2 What we propose

Read the probability from the model's output distribution instead.

At each generated token position the model defines a probability distribution
over its vocabulary $V$. Let $t^\*$ be the position at which the verdict word
begins, let $A_1 \subset V$ be the token strings spelling `correct` and
$A_0 \subset V$ those spelling `incorrect`. Define

$$p^{\text{raw}} \;=\; \frac{\sum_{v \in A_1} \Pr(v \mid \text{prompt})}{\sum_{v \in A_1} \Pr(v \mid \text{prompt}) \;+\; \sum_{v \in A_0} \Pr(v \mid \text{prompt})}$$

This is the model's probability of `correct` **conditional on emitting one of the
two verdicts**, which under the constrained verdict-only prompt it does with
probability ≈ 1. Renormalising is what makes $p^{\text{raw}}$ a probability over
$Y$ rather than over the whole vocabulary.

This belongs to the information-based family. It is the binary-classification
analogue of **Maximum Sequence Probability (MSP)**, the simplest information-based
baseline, defined in the same benchmark as $U_{\text{MSP}}(x) = 1 - \Pr(y \mid x)$.
The benchmark reports MSP as "a very strong and robust baseline across all tasks",
and finds information-based methods best for *short* outputs specifically — our
output is fourteen tokens.

One clarification, to avoid overclaiming. In that benchmark, sequence probability
is a *proxy* for the quality of a free-form generation. Our setting is different
and simpler: the model emits a single classification token, so the renormalised
distribution is not a proxy for anything — it is the model's distribution over
the label itself.

**Measured example** (GLM-5, real corpus executions, all emitting `correct` /
`high`, hence all mapped to 0.95 by the table):

| execution | $p^{\text{raw}}$ | table score |
|---|---|---|
| 0/3 | 0.99997 | 0.95 |
| 0/1 | 0.99720 | 0.95 |
| 0/0 | 0.98409 | 0.95 |
| 0/2 | 0.93244 | 0.95 |
| 0/5 | 0.70576 | 0.95 |
| 0/4 | 0.67916 | 0.95 |

### 2.3 Availability constraint

Information-based methods require access to the output distribution. The
benchmark calls methods that need this **white-box**, and methods needing only
the generated text **black-box**. Measured against our provider:

| model | route | token log-probabilities |
|---|---|---|
| GLM-5 | chat completions | returned, with top-*k* alternatives |
| gemma-4-26b / 31b / e2b | responses | parameter accepted, **empty array returned** |

If a model cannot be made to return them, the black-box substitute in the same
benchmark is **LabelProb**: sample $K$ outputs and estimate the label probability
by relative frequency,
$\hat{\Pr}(y \mid x) = \frac{1}{K}\sum_{i=1}^{K} \mathbb{1}(y^{(i)} = y)$.
This estimates the same quantity at $K\times$ the inference cost, with resolution
limited to multiples of $1/K$. It is the honest fallback; reverting to verbalized
confidence is not.

---

## 3. Layer 2 — calibration

### 3.1 Definition

A probabilistic forecast $p$ for a binary outcome $Y$ is **calibrated** (equivalently
**reliable**) if

$$\mathbb{E}[Y \mid p = v] = v \qquad \text{for every } v \text{ in the range of } p$$

In words: among all cases where the forecast said 0.7, the outcome occurs 70% of
the time. Calibration is a property of the *forecast–outcome joint distribution*,
not of the model's internals, and it is not implied by accuracy: a model can rank
cases perfectly and still be systematically overconfident.

Raw language-model probabilities are generally not calibrated, so $p^{\text{raw}}$
must be mapped before it is interpreted as a belief.

### 3.2 The map

Fit a monotone non-decreasing function $\hat{c}$ on a held-out calibration set
$\{(p^{\text{raw}}_i, y_i)\}$ by **isotonic regression** — the least-squares fit
subject to a monotonicity constraint,

$$\hat{c} \;=\; \arg\min_{f \text{ non-decreasing}} \sum_i \bigl(y_i - f(p^{\text{raw}}_i)\bigr)^2$$

solved exactly by the **Pool Adjacent Violators Algorithm (PAVA)**. PAVA merges
adjacent order-violating points into blocks and assigns each block its mean,
producing a piecewise-**constant** step function.

Monotonicity is the entire parametrisation: there is no bin count, bandwidth, or
smoothing parameter to choose. That is what makes it parsimonious.

Because PAVA is piecewise constant, every case inside a pooled block receives an
identical fitted value — so cases that were strictly ordered in $p^{\text{raw}}$
become tied. **Centered Isotonic Regression (CIR)** (Oron & Flournoy, 2017)
removes the ties: it collapses each flat block to a single point at the weighted
centre of the block's $x$-values and linearly interpolates between those points,
yielding a strictly monotone piecewise-**linear** function. Vashurin et al.
recommend CIR for exactly this reason — a calibration map that preserves the
ranking cannot degrade any ranking-based metric.

Use PAVA when only the calibrated value is wanted; use CIR when ties would be
destructive downstream.

### 3.3 Why this replaces the table

The six-cell table is a calibration map with six assigned values and no fitting
procedure. Isotonic regression is a calibration map with zero assigned values,
fitted to data. The proposal removes six free constants and adds no hyperparameters.

---

## 4. Layer 3 — pooling evidence into a statement belief

### 4.1 What is deployed: a noisy-OR

In Bayesian networks, the **noisy-OR** model expresses a binary effect produced
by several independent causes, each of which may independently fail to produce
it. If cause $i$ fails with *inhibition probability* $q_i$, then

$$\Pr(\text{effect}) \;=\; 1 - \prod_i q_i$$

INDRA instantiates this over sources. Each source $s$ carries two parameters:

- $\text{syst}_s$, a **systematic error rate** — the probability the source is
  wrong in a way that repetition cannot fix;
- $\text{rand}_s$, a **random error rate** — the probability that any single
  piece of its evidence is an independent one-off error.

With $n_s$ pieces of evidence from source $s$, the source's inhibition
probability is $\text{syst}_s + \text{rand}_s^{\,n_s}$, and

$$\text{belief} \;=\; 1 - \prod_s \bigl(\text{syst}_s + \text{rand}_s^{\,n_s}\bigr)$$

Note this is the **additive** form as implemented in INDRA's `SimpleScorer`, not
the conditional form $\text{syst} + (1-\text{syst})\cdot\text{rand}^{\,n}$ that
appears in Gyori et al. (2017); the discrepancy is documented in `noise_model.py`.

Source parameters re-estimated on a 9,342-curation benchmark as
$\text{rand} = 1 - \text{accuracy} - \text{syst}$:

| source | rand | syst | belief from one evidence |
|---|---|---|---|
| reach | 0.462 | 0.05 | 0.488 |
| sparser | 0.516 | 0.05 | 0.434 |
| medscan | 0.481 | 0.05 | 0.469 |
| trips | 0.077 | 0.05 | 0.873 |
| rlimsp | 0.056 | 0.05 | 0.894 |
| signor | 0.049 | 0.01 | 0.941 |
| biogrid | 0.010 | 0.01 | 0.980 |

### 4.2 Why the noisy-OR is the wrong form here

The noisy-OR answers: *what is the probability that at least one source
succeeded?* Its value is **monotone non-decreasing in evidence count**. For
`reach`:

| $n$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| belief | 0.488 | 0.737 | 0.851 | 0.904 | 0.929 | 0.940 |

Adding evidence can only raise belief. **The form has no representation for
disconfirming evidence.** This is why the deployed code must physically *delete*
rejected reads before scoring, and drop a source entirely when none of its
evidence survives — the "hard gate" is a workaround for a functional form that
cannot express a negative.

The cost is measured elsewhere in this repository: the noisy-OR assigns identical
65% → 95% trajectories to sources whose measured accuracy spans 12%–80%, and
`reach` **anti**-correlates with correctness (48% → 31%). Those are properties of
the functional form, not of the priors.

### 4.3 The proposal: pooling weights of evidence in log-odds

Define the **odds** of an event with probability $p$ as $p/(1-p)$, and the
**log-odds** or **logit** as

$$\operatorname{logit}(p) \;=\; \log\frac{p}{1-p}$$

with inverse the **logistic** (or **sigmoid**) function
$\sigma(x) = 1/(1+e^{-x})$.

Let $\pi = \Pr(Y=1)$ be the **prior** — the base rate of correct extractions
before reading anything. For a single read $r$, Bayes' theorem in odds form is

$$\frac{\Pr(Y{=}1 \mid r)}{\Pr(Y{=}0 \mid r)} \;=\; \underbrace{\frac{\Pr(r \mid Y{=}1)}{\Pr(r \mid Y{=}0)}}_{\text{likelihood ratio}} \times \frac{\Pr(Y{=}1)}{\Pr(Y{=}0)}$$

Taking logarithms turns the product into a sum:

$$\operatorname{logit}\bigl(\Pr(Y{=}1 \mid r)\bigr) \;=\; \operatorname{logit}(\pi) \;+\; \underbrace{\log \frac{\Pr(r \mid Y{=}1)}{\Pr(r \mid Y{=}0)}}_{W(r)}$$

$W(r)$ is the **log-likelihood ratio**, also called the **weight of evidence**
(I. J. Good, 1950). It is the amount the read moves belief on the log-odds scale,
and it is a property of the reader conditional on the truth — it does not depend
on the base rate.

Rearranging gives the identity we use:

$$W(r) \;=\; \operatorname{logit}\bigl(\hat{c}(r)\bigr) - \operatorname{logit}(\pi) \tag{$\ast$}$$

where $\hat c(r)$ is the calibrated posterior from read $r$ alone (§3). That is:
**a calibrated posterior can be converted into a weight of evidence by
subtracting the prior on the log-odds scale.**

If reads are **conditionally independent given $Y$** — that is,
$\Pr(r_1,\dots,r_n \mid Y) = \prod_j \Pr(r_j \mid Y)$ — their likelihood ratios
multiply, so their weights of evidence add:

$$\operatorname{logit}\bigl(\Pr(Y{=}1 \mid r_1,\dots,r_n)\bigr) \;=\; \operatorname{logit}(\pi) + \sum_{j=1}^{n} \Bigl[\operatorname{logit}\bigl(\hat c_j\bigr) - \operatorname{logit}(\pi)\Bigr]$$

This is the standard **naive Bayes** combination rule. ("Naive" names the
conditional-independence assumption, which is an approximation here, not a
claim.) The final belief is recovered with the logistic function.

**Why this is the right form for the problem.** A read that says *incorrect*
yields $\hat c_j < \pi$, hence a negative weight of evidence, and belief falls.
No deletion rule, no source removal, no special case. The estimator handles
confirming and disconfirming evidence by the same arithmetic.

### 4.4 Correlated reads within a source

Conditional independence is implausible for two evidences read by the *same*
reader: they share the reader's systematic biases. Independent summation would
double-count.

The deployed system averages within a source and sums across sources:

$$\operatorname{logit}(\text{belief}) \;=\; \operatorname{logit}(\pi) \;+\; \sum_{\text{sources } s} \ \underbrace{\frac{1}{n_s}\sum_{j \in s} W_j}_{\text{mean within source}}$$

Averaging is equivalent to a weight $w_s = 1/n_s$: the source counts once no
matter how much evidence it supplies. This is the *maximally conservative*
correlation correction — it assumes reads within a source are perfectly
correlated, contributing no independent information.

We propose keeping it, stating it as an explicit assumption, and testing it
against alternatives ($w_s = 1/\sqrt{n_s}$, which assumes partial correlation; or
a fitted $w_s$) on held-out data. It should be a measured choice, not an
inherited one.

---

## 5. The proposal is a strict generalization of the deployed estimator

The deployed system also aggregates in log-odds, but with a per-verdict weight
of evidence derived from a $2\times2$ **confusion matrix** — the cross-tabulation
of model verdict against curator label:

|  | gold correct | gold incorrect |
|---|---|---|
| model says correct | $cc$ | $ci$ |
| model says incorrect | $ic$ | $ii$ |

From these counts, define the **sensitivity** (true positive rate)
$\Pr(\text{confirm} \mid Y{=}1) = cc/(cc+ic)$ and the **false positive rate**
$\Pr(\text{confirm} \mid Y{=}0) = ci/(ci+ii)$. These are conditioned on the truth,
so they are properties of the reader and free of the base rate. The two weights
of evidence follow directly:

$$W_{\text{confirm}} = \log\frac{\text{sensitivity}}{\text{false positive rate}}, \qquad W_{\text{reject}} = \log\frac{1-\text{sensitivity}}{1-\text{false positive rate}}$$

For the deployed `gemma_remote` profile ($cc{=}704$, $ci{=}157$, $ic{=}97$,
$ii{=}646$): sensitivity 0.8789, false positive rate 0.1955, giving
$W_{\text{confirm}} = +1.5030$ and $W_{\text{reject}} = -1.8936$, with
$\operatorname{logit}(\pi) = -0.0025$.

Inverting $(\ast)$ recovers the only two posteriors this estimator can express
per read:

$$\hat c_{\text{confirm}} = \sigma(-0.0025 + 1.5030) = 0.8177, \qquad \hat c_{\text{reject}} = \sigma(-0.0025 - 1.8936) = 0.1306$$

**So the deployed estimator is the proposed estimator restricted to two atoms.**
The six-cell table's apparent granularity is not present in the statement belief;
only the verdict survives.

There is one exception. The deployed code applies a floor to confirming reads,

$$\ell(\text{correct}) = \max\bigl(W_{\text{confirm}},\; \operatorname{logit}(1 - (\text{syst}_s + \text{rand}_s))\bigr)$$

so that a confirmation from a highly reliable source is not diluted by a generic
reader profile. The second term is a **reliability logit**, not a likelihood
ratio, so combining them under a maximum is not an application of Bayes' rule —
the code documents this explicitly and calls the result a hybrid.

Running both estimators on identical reads:

| reads | deployed | proposed | difference |
|---|---|---|---|
| reach:+ | 0.817654 | 0.817654 | 0 |
| reach:+, reach:+ | 0.817654 | 0.817654 | 0 |
| reach:+, sparser:+ | 0.952735 | 0.952735 | 0 |
| reach:− | 0.130552 | 0.130552 | 0 |
| reach:+, sparser:− | 0.402980 | 0.402980 | $10^{-16}$ |
| reach:+, reach:−, trips:+ | 0.849409 | 0.786717 | −0.063 |

The two agree **exactly** wherever the floor does not bind, and differ only where
it does. `trips` has reliability logit $+1.9277 > W_{\text{confirm}} = +1.5030$,
so the floor fires; `reach` ($-0.0480$) and `sparser` ($-0.2655$) never do.

This makes the migration defensible: the proposal generalizes the deployed
estimator from two atoms to a continuum, and removes precisely the one element
the implementation itself identifies as non-Bayesian. Existing validation carries
over wherever the floor is inactive.

---

## 6. Source reliability as an empirical question

The floor exists to stop a mediocre reader from overriding a highly reliable
source. But the reader is judging *whether this sentence supports this claim* —
which is the very failure mode the source prior summarises. To that extent the
read **screens off** the source: once you know what the sentence says, knowing
which reader extracted it adds less.

How much less is measurable. Fit nested models on held-out gold and compare:

- **M0** — read only: $\operatorname{logit}(\text{belief}) = \operatorname{logit}(\pi) + \sum_s \frac{1}{n_s}\sum_j\bigl[\operatorname{logit}(\hat c_j) - \operatorname{logit}(\pi)\bigr]$
- **M1** — M0 $+\ \beta \cdot \operatorname{logit}(1 - (\text{syst}_s + \text{rand}_s))$
- **M2** — M1 $+$ the INDRA noisy-OR belief as an additional feature

Report each model's held-out log-loss and Brier score, and the fitted $\beta$
with its confidence interval. If $\beta$ is indistinguishable from zero, the
source term is dropped — parsimony *demonstrated* rather than assumed. If it is
not, we have a fitted coefficient in place of a hard-coded maximum. Both outcomes
are reportable; the current floor is neither fitted nor tested.

---

## 7. Evaluation

Three distinct properties, measured separately. Conflating them is the most
common error in this area.

**Discrimination** — can the score rank correct above incorrect? Measured by
**AUROC** (the probability a randomly chosen correct case scores above a randomly
chosen incorrect one) and **average precision** (the area under the
precision–recall curve, computed over distinct score thresholds so that tied
scores collapse to a single point).

**Calibration** — do the numbers mean what they say (§3.1)?

**Overall forecast quality** — measured by a **proper scoring rule**, i.e. a
scoring rule whose expected value is optimised by reporting one's true belief.
The **Brier score** $\text{BS} = \frac{1}{N}\sum_i (p_i - y_i)^2$ is strictly
proper for binary outcomes; **log-loss** is the other standard choice.

The Brier score decomposes (Murphy, 1973) as

$$\text{BS} \;=\; \underbrace{\text{REL}}_{\text{reliability}} - \underbrace{\text{RES}}_{\text{resolution}} + \underbrace{\text{UNC}}_{\text{uncertainty}}$$

where reliability measures miscalibration (lower is better), resolution measures
how far forecasts move from the base rate (higher is better), and uncertainty
$\bar y(1-\bar y)$ depends only on the data. This decomposition separates
calibration from discrimination inside a single proper score.

Two practical notes.

*The classical decomposition is computed over bins, and is therefore
partition-dependent.* The **CORP** decomposition (Dimitriadis, Gneiting & Jordan,
2021) replaces binning with PAVA, giving miscalibration (MCB), discrimination
(DSC) and uncertainty (UNC) with no bin parameter. Since we already use isotonic
regression for calibration (§3.2), using the same estimator for evaluation is
free and removes a hyperparameter. We measured the two to agree to ≈0.001 on this
data, so this changes little numerically and removes an arbitrary choice.

*Expected Calibration Error (ECE)* — the bin-weighted mean gap between mean
forecast and observed frequency — is *not* partition-invariant in general, and
should be reported with its partition stated. In the special case of the deployed
six-cell table it happens to be exact rather than approximate, because the
attainable scores are 0.15 apart and the deployed bin edges isolate each one; we
verified ECE is identical to six decimals under every equal-width partition with
≥7 bins. Once scores are continuous this exactness disappears, and the bin-free
estimator becomes the appropriate choice.

Finally, thresholded metrics. Reporting error-detection F1 at a chosen cut-off
$\tau$ requires selecting $\tau$, and Vashurin et al. note that such choices have
"been quite arbitrary in the literature", recommending instead threshold-free
**prediction-rejection curves** summarised by the **Prediction Rejection Ratio**

$$\text{PRR} = \frac{\text{AUC}_{\text{unc}} - \text{AUC}_{\text{rnd}}}{\text{AUC}_{\text{oracle}} - \text{AUC}_{\text{rnd}}}$$

which measures how much of the achievable quality gain a score realises when
low-confidence cases are progressively rejected. Our shipped gate uses a
threshold-selected F1 as its lead metric; PRR is the standard threshold-free
complement.

---

## 8. Assumptions, stated

1. **Conditional independence across sources.** Required for weights of evidence
   to add. Different readers on the same sentence share the sentence, so this is
   an approximation. It is the same assumption INDRA's noisy-OR already makes.
2. **Perfect correlation within a source** ($w_s = 1/n_s$). Conservative, and
   testable (§4.4).
3. **The calibration map transfers.** $\hat c$ is fitted on one gold set and
   applied elsewhere; if the new corpus differs in difficulty or composition, the
   map may not hold. Requires a held-out calibration set and periodic refitting.
4. **The prior $\pi$ is the operating base rate.** The fitted profiles use the
   evidence-pair base rate of the calibration corpus (0.4994). A deployment
   corpus with a different prevalence needs $\pi$ restated; because the estimator
   is a genuine posterior once the floor is removed, this is a clean substitution
   rather than a global score shift.
5. **The renormalisation in §2.2 is valid** only while the model reliably emits
   one of the two verdict tokens. Measured at ≈99.9% under the verdict-only
   prompt; the residual is retried and, if it persists, recorded as an error
   rather than assigned a score.

---

## 9. Summary of the change

| | deployed | proposed |
|---|---|---|
| per-evidence score | verbalized label → 6 assigned constants | renormalised token probability |
| distinct values per read | 2 (0.8177 / 0.1306) | continuous on $(0,1)$ |
| calibration | none (the table *is* the map) | isotonic / CIR, 0 hyperparameters |
| pooling | log-odds sum, with a non-Bayesian floor | log-odds sum of weights of evidence |
| disconfirming evidence | reads deleted, sources dropped | negative weight, no special case |
| source reliability | hard-coded maximum | fitted coefficient, or dropped if not significant |
| assigned constants removed | — | 6 grid values, the floor (the 0.5 parse fallback was already removed in `a10df62`; an unparseable reply is typed absence, not a fabricated midpoint) |

The estimator is not new — it is naive Bayes in log-odds, which is what the
deployed system already approximates. What changes is that the per-read input
becomes a measurement rather than a lookup, and the one step that was not derived
from Bayes' rule is removed.

---

### References

- Good, I. J. (1950). *Probability and the Weighing of Evidence.* Griffin. — weight of evidence.
- Murphy, A. H. (1973). A new vector partition of the probability score. *J. Appl. Meteorol.* 12:595–600. — reliability/resolution/uncertainty.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.* — noisy-OR.
- Oron, A. P. & Flournoy, N. (2017). Centered isotonic regression. *Stat. Biopharm. Res.* 9:258–267.
- Dimitriadis, T., Gneiting, T. & Jordan, A. I. (2021). Stable reliability diagrams for probabilistic classifiers. *PNAS* 118(8). — CORP.
- Gyori, B. M. et al. (2017). From word models to executable models of signaling networks. *Mol. Syst. Biol.* 13:954. — INDRA belief model.
- Vashurin, R. et al. (2025). Benchmarking uncertainty quantification methods for LLMs with LM-Polygraph. *TACL* 13:220–248.
