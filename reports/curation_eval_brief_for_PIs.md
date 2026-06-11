# Can an AI reliably catch errors in text-mined biomedical relationships?
### A head-to-head on expert-curated human gold — a non-technical brief

**The short version**

- We tested two AI models as *checkers* of automatically-extracted biomedical statements (e.g. "*MTOR phosphorylates IRS1*"), grading them against **expert human curations** on a deliberately balanced test set.
- A large general model (**gemma, 26B parameters**) catches **80% of the genuinely-wrong extractions**; a small specialized medical model (**MedPsy, 4B**) catches **71%** — both at the same false-alarm rate. gemma is meaningfully and statistically better, but MedPsy is competitive at **one-sixth the size**.
- The models' weaknesses are *specific and biological* (confusing "represses transcription" with "inhibits activity"; accepting two co-mentioned proteins as interacting partners). And — the most useful finding — the small model usually **already knows the right answer and talks itself out of it**, which means much of the gap is fixable without a bigger, costlier model.

---

### Why this matters

Tools like INDRA read the literature at scale and extract structured relationships — millions of "A regulates B" statements assembled from many NLP readers. These power pathway maps, hypothesis generation, and knowledge bases. But readers make mistakes: they mis-identify genes, mistake co-mention for interaction, or garble the direction of an effect. **Someone has to check.** Human curation is the gold standard but doesn't scale. The question is whether an AI can do the checking reliably enough to trust — and if so, which one, and how cheaply.

### What we did (and why you can trust the numbers)

**The gold standard.** We drew our test set from the **INDRA assembly benchmark** — thousands of (statement, evidence) pairs each labeled correct or incorrect by expert curators (Bachman & Gyori), with the *specific error type* recorded (wrong gene, no real relation, activity-vs-amount confusion, etc.).

**We balanced it 50/50.** A naive test set is ~90% correct extractions, and on such a set "accuracy" is dominated by the easy cases — an over-lenient model looks great. An earlier spot-check on that kind of skewed data showed the two models in a dead heat; **that tie was an artifact.** We built a balanced set — **803 correct and 803 incorrect pairs** — so the headline measures the thing that matters: *catching the wrong ones.*

**We eliminated leakage.** Every pair either model had previously seen as a worked example, and every pair from any prior test set, was excluded. We also matched the two classes on statement type, so "Complex is harder than Phosphorylation" can't masquerade as a model difference.

**We graded on error-detection, not accuracy.** The headline number is the model's skill at flagging the curator-flagged *wrong* extractions, reported as **precision** (of the ones it flags, how many are truly wrong) and **recall** (of the truly-wrong ones, how many it catches).

### What we found

| | Accuracy | Catches real errors (recall) | False-alarm control (precision) | Overall error-detection |
|---|---|---|---|---|
| **gemma (26B, general)** | **84.1%** | **80%** | 87% | **0.835** |
| **MedPsy (4B, medical)** | 80.4% | 71% | 87% | 0.785 |

The two flag wrong extractions with **equal precision** — when either says "this is wrong," it's right ~87% of the time. The difference is **recall**: gemma simply *catches more* of the real errors. The gap is statistically robust (a paired test gives p = 0.0001), not noise.

**Where the gap lives — and it's biology, not vibes.** gemma's advantage concentrates in three error types:
- **Activity vs. amount** (gemma 79% vs MedPsy 56%): "*X transcriptionally represses Y*" or "*degrades Y*" is a change in **amount**, not an inhibition of **activity** — a distinction INDRA cares about. The small model repeatedly misses it.
- **No real relation** (88% vs 74%): two proteins co-mentioned in a sentence — *"c-Myb and GATA-3 bound to their respective DNA sites"* — are not interacting *with each other.* The small model accepts the pairing anyway.
- **Wrong relation** (88% vs 77%): the effect is real but the *type* of relationship is mis-stated.

### The deeper result: it's commitment, not knowledge

We read the models' full reasoning on every error. The striking pattern in the small model:

- In **44% of its mistakes, MedPsy reached the correct judgment in its own words** — *"the evidence does not explicitly say they form a complex"* — and then wrote a *"but in many contexts this is a standard mechanism…"* sentence that **reversed it**.
- It shows a **systematic leniency bias**: under uncertainty its default is "correct," and it will **invent a supporting fact** to get there (in one case asserting a gene alias that doesn't exist) in ~half of its errors.
- Only **~6%** of its errors look like a genuine knowledge ceiling. The model that gets these right uses the *same* perception — it just **reaches the disqualifying fact and stops, instead of arguing past it.**

**Implication — and we tested it.** The headline gap is largely a *calibration / commitment* problem, not a capacity problem. So we changed nothing about the model and only restructured **how it reports its judgment**: we required it to write down the single strongest reason the extraction might be wrong *before* committing to a verdict, plus a "guilty until proven innocent" stance (incorrect unless the text explicitly states the relationship). The result: the small model's error-catching jumped from **71% to 78% recall** (overall error-detection 0.785 → 0.818), **closing about two-thirds of the gap to the large model — with the identical 4B model.** It simply stopped arguing past its own findings once forced to state them first. (Statistically significant, p = 0.008; a modest, expected uptick in false alarms came with the stricter stance.) This is the practical headline: *much of the apparent "you need a bigger model" gap is recoverable through prompt and output design.*

### What this means for you

- **For automated error-checking today, the larger general model is the more reliable referee** — it catches four-in-five genuine errors at a low false-alarm rate.
- **A small specialized model is a credible cost/throughput option** (one-sixth the size, competitive accuracy), and most of its shortfall appears to be addressable.
- **Methodology matters as much as the model.** Judge these tools on a *balanced, leakage-free* set using *error-detection*, not accuracy on naturally-skewed data — otherwise a lenient model will look deceptively good (as it did in our first pass).

### Limitations

This is one corpus (the INDRA assembly benchmark) and one scoring architecture. The curations themselves are not infallible — our adversarial re-check found a small fraction (~5–10%) where the curator's label is debatable, and we excluded those from the error analyses. Recall and precision confidence intervals for the two models overlap at the *per-category* level even though the overall paired comparison is significant. The "fixable without scaling" result (the prompt-restructuring that recovered two-thirds of the gap) is from a single intervention on this set; the stricter stance trades a little precision for recall, and the right operating point should be tuned on a separate calibration set before deployment.

---

*Test set: 1,606 expert-curated (statement, evidence) pairs, balanced 803/803, fresh and de-contaminated. Both models scored identically on the same evidence and graded against the human curations.*
