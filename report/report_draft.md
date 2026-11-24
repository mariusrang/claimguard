---
title: "ClaimGuard: Evidence-Based Claim Verification with Calibrated Abstention"
author: "Marius Rang — NLP final project, topic 2 (Fact Verification with Evidence Retrieval)"
geometry: margin=2.2cm
fontsize: 10.5pt
---

> **Draft status.** Sections 1–5 are written with the real numbers from `results/`.
> Still to do: Section 6 Discussion and the conclusion paragraph (Marius), the
> optional contrast-set hand check in 3.4. Remove this box
> before exporting to PDF.

## Abstract

Language models produce fluent answers that are not always supported by their
sources. We present ClaimGuard, a pipeline that verifies a claim against a document
collection and returns SUPPORTED, REFUTED or NOT ENOUGH INFO with the evidence and a
calibrated confidence. The pipeline combines hybrid retrieval (BM25 and a dense
encoder fused with Reciprocal Rank Fusion), cross-encoder reranking, a DeBERTa-v3
verifier fine-tuned on SciFact, multi-evidence aggregation and temperature scaling.
On the SciFact test set the full system reaches 0.693 ± 0.010 macro-F1 (3 seeds),
against 0.609 for a zero-shot NLI baseline and 0.671 for a 7B open-source LLM judge
that is 26 times slower. With gold evidence, macro-F1 rises to 0.743: retrieval
causes some errors, but the verifier is the main bottleneck. On sentences written by
LLMs (LLM-AggreFact) the verifier reaches a mean balanced accuracy of 0.709. After
temperature scaling the expected calibration error drops from 0.260 to 0.141, and
with a threshold of 0.8 the alerts are right 90.5% of the time for a coverage of 70%.
A contrast set shows that the verifier catches 97% of negation edits but only 30% of
changed numbers. A manual analysis of 110 errors finds that the most frequent failure
is treating evidence on the right topic as proof. The system ships with a demo, a
Docker image and an MCP server with an audit log.

<!-- 150-200 words once the numbers are in -->

## 1. Introduction

Retrieval-augmented generation is now the standard way to make a language model
answer from a company's own documents. The remaining problem is trust: the answer
reads well whether or not each sentence is backed by the documents, and checking it
by hand removes most of the benefit. Fact verification with evidence retrieval is the
research task behind this problem. Given a claim, a system must find evidence in a
corpus and decide if the evidence supports the claim, refutes it, or is not enough to
decide.

Recent work has shown that small specialised models can check claims against
documents as well as large language models for a fraction of the cost (Tang et al.,
2024). We do not try to show this again. We look at three questions that matter when
such a verifier is actually deployed and that are less often reported together:

1. **Can the confidence be trusted?** A verifier that is wrong with 95% confidence is
   worse than one that says "I am not sure". We calibrate the verifier and measure
   what abstention buys.
2. **Does it catch the errors that hurt?** A changed amount, a removed negation or an
   opposite word leaves almost all the words of a claim unchanged. We build a
   contrast set for these edits and a small French banking test set.
3. **Where do the errors come from?** With retrieval and verification evaluated
   separately, and an oracle experiment with gold evidence, we can say which stage
   deserves the next hour of work.

Our contributions are: (i) a complete, tested pipeline in which every stage can be
swapped and measured alone; (ii) an evaluation with five baselines, six ablations,
confidence intervals and significance tests; (iii) a calibration and abstention
study with a metric that non-technical readers understand, the precision of the
alerts; (iv) three stress tests; (v) an evaluation of the same pipeline on sentences written
by LLMs (LLM-AggreFact); (vi) an analysis of 110 errors; and (vii) a
deployable tool: demo, Docker image, INT8 model and an MCP server with an audit log.

One limit frames everything that follows: the system measures whether a claim is
supported **by the documents**. It does not know if the documents are right.

## 2. Related Work

**Fact verification datasets.** FEVER (Thorne et al., 2018) defined the task at scale
with 185k claims verified against Wikipedia and a score that requires both the right
label and the right evidence. SciFact (Wadden et al., 2020) moved the task to
scientific claims verified against research abstracts, with sentence-level
rationales; it is small, which makes it a good test of data-efficient methods.
Climate-FEVER (Diggelmann et al., 2020) collected real-world climate claims and is
often used as an out-of-domain test. Schuster et al. (2019) showed that FEVER claims
alone, without evidence, already predict the label well above chance, which is why we
include a claim-only baseline.

**Retrieval.** BM25 (Robertson and Zaragoza, 2009) remains a strong lexical baseline.
Dense retrieval with bi-encoders (Karpukhin et al., 2020) matches meaning instead of
words; general-purpose embedding models such as BGE (Xiao et al., 2023) now work well
without task-specific training. Lexical and dense rankings fail on different queries,
and Reciprocal Rank Fusion (Cormack et al., 2009) combines them using ranks only.
Cross-encoders (Nogueira and Cho, 2019) read the query and the passage together and
are the usual second stage.

**Verification.** Verification is close to natural language inference, and models
pre-trained on NLI data transfer well (Laurer et al., 2024). DeBERTa-v3 (He et al.,
2023) is a strong encoder for this kind of sentence-pair task. On SciFact, MultiVerS
(Wadden et al., 2022) predicts the label and the rationales jointly from the full
abstract and uses weak supervision from other datasets.

**Checking LLM outputs.** FActScore (Min et al., 2023) splits a generated text into
atomic facts and checks each one. SummaC (Laban et al., 2022) revisited NLI models
for inconsistency detection in summaries. MiniCheck (Tang et al., 2024) trains small
models that reach the accuracy of GPT-4-class judges on the LLM-AggreFact benchmark
at a much lower cost. Our verifier belongs to the same family; our focus is on
calibration, stress tests and deployment and we do not claim a better checker.

**Calibration and robustness.** Modern networks are over-confident, and temperature
scaling is a simple and effective fix (Guo et al., 2017). Selective classification
studies the trade-off between coverage and risk when a model may abstain (Geifman and
El-Yaniv, 2017). Contrast sets (Gardner et al., 2020) test a model with small,
meaning-changing edits of test examples.

## 3. Methodology

### 3.1 Task and data

A claim *c* must be labelled SUPPORTED, REFUTED or NEI using a corpus of abstracts.
We use SciFact: 5,183 abstracts and 1,109 labelled claims. The official test labels
are hidden, so the official dev set (300 claims) is our test set and we split the
official train set 85/15 into train and dev, stratified by label. A claim is
SUPPORTED or REFUTED if it has gold evidence with that label, NEI otherwise. This is
a claim-level, three-way setup, so our numbers are not comparable with the official
abstract-level leaderboard.

| Split | Claims | SUPPORTED | REFUTED | NEI | Note |
| --- | --- | --- | --- | --- | --- |
| train | 687 | 282 | 147 | 258 | 85% of the official train set |
| dev | 122 | 50 | 26 | 46 | 15% of the official train set, stratified, seed 13 |
| test | 300 | 124 | 64 | 112 | = official dev set |

Corpus: 5,183 abstracts, 8.9 sentences per abstract on average. Claims are about 13
words long; a claim with evidence has 1.1 evidence abstracts on average.

### 3.2 Pipeline

**First-stage retrieval.** Three interchangeable retrievers over title + abstract:
(a) BM25 with k1 = 1.5 and b = 0.75, implemented with an inverted index; (b) a dense
bi-encoder (BGE-small-en-v1.5, CLS pooling, normalised vectors, exact cosine search);
(c) a hybrid that fuses the two top-50 lists with RRF,
score(d) = Σ 1 / (60 + rank(d)).

**Reranking and sentence selection.** A MiniLM cross-encoder trained on MS MARCO
scores the 50 candidates and keeps the top k (k = 5 by default). The same model then
scores the sentences of each kept abstract and keeps the best 3, in reading order.
These sentences are the evidence given to the verifier.

**Verification.** DeBERTa-v3-base reads (evidence, claim) as an NLI pair and outputs
three logits. We start from a public checkpoint trained on MNLI, FEVER-NLI and ANLI
and fine-tune it on pairs built from SciFact: gold rationale sentences with the claim
label; the selected sentences of cited abstracts for NEI claims; and hard negatives,
one abstract retrieved by BM25 that is not gold evidence, labelled NEI. Training uses
AdamW, linear warmup over 10% of the steps then linear decay, gradient clipping at
1.0, and keeps the epoch with the best dev macro-F1.

**Aggregation.** Each of the k documents gives a probability vector. We compare a
soft majority vote (the mean of the k vectors), the single most decisive document, and
a noisy-OR:
P(S) = 1 − Π(1 − pᵢ(S)), P(R) = 1 − Π(1 − pᵢ(R)), P(NEI) = Π pᵢ(NEI), renormalised.

**Calibration and abstention.** Passage logits are divided by a temperature T before
the softmax and the aggregation. T is fitted on dev by minimising the negative
log-likelihood of the *claim-level* probabilities, which is the confidence the user
sees; fitting it on the passage pairs instead made the claim-level ECE worse in our
first CPU runs. The claim-level confidence is the largest aggregated probability. Under a
threshold τ the system answers UNCERTAIN. τ is the smallest value that reaches an
alert precision of 0.9 on dev.

### 3.3 Baselines

(1) Majority class. (2) Claim-only: TF-IDF and logistic regression on the claim text,
no evidence. (3) Lexical rules: word overlap, number mismatch and negation mismatch.
(4) Minimal system: BM25, no reranker, the NLI checkpoint without fine-tuning.
(5) LLM judge: an open-source instruction model (Qwen2.5-7B, 4-bit) reads the same
evidence and answers with one letter; we read the logits of the three letters.

### 3.4 Stress tests

**Contrast set.** From each SUPPORTED test claim and its gold evidence we generate
edited claims with one rule: change a number, add or remove a negation, or swap a
direction word for its opposite. We report the share of edited claims that are no
longer predicted SUPPORTED and the consistency (right on the original and reacts to
the edit). This gives 125 edited claims: 20 number edits, 64 negation edits and 41
direction edits. [[TODO optional: share of a hand-checked sample of 50 where the edit
really breaks the support]]

**French banking set.** 60 claims on 20 invented documents (20 per label); REFUTED
claims change an amount, a rate, a duration, a threshold, a date, a direction or a
negation. We compare the English verifier, a multilingual NLI model and the rules.

**Climate-FEVER.** 1,381 claims (DISPUTED removed), five given evidence sentences
each, verifier and aggregation only, no fine-tuning on this domain.

### 3.5 Checking LLM answers

An answer is split into sentences; questions, greetings and very short sentences are
removed, and each remaining sentence is a claim. The source document is cut into
chunks of 6 sentences; BM25 and the cross-encoder pick the best chunks, then the
verifier and the aggregation run as above. We evaluate on LLM-AggreFact (Tang et al.,
2024), which gathers summaries and RAG answers written by several LLMs, each sentence
labelled supported or not by humans. The task is binary, so SUPPORTED is compared
with REFUTED or NEI, and we report balanced accuracy per sub-dataset and its mean,
the benchmark's own metric. Because of the GPU budget we use 100 claims per
sub-dataset; numbers are therefore not directly comparable with the full leaderboard.

### 3.6 Evaluation protocol

Every setting is chosen on dev and never on test: the number of evidence passages
k ∈ {1, 3, 5, 10}, the aggregation strategy and, for the main system, the temperature
(fitted for each setting on the claim-level probabilities). Baselines go through the
same selection with T = 1.
Retrieval is evaluated alone on claims that have evidence: recall@k, precision@k,
MRR. End-to-end we report accuracy, macro-F1, a FEVER-style score (right label and at
least one gold abstract among the evidence), and evidence-sentence F1. Calibration is
measured with ECE (10 bins), the risk-coverage curve and its area (AURC), and alert
precision and recall. The verifier is trained with 3 seeds. Every headline number
has a 95% bootstrap interval (1,000 resamples); systems are compared with McNemar's
exact test, retrievers with a paired permutation test on recall@5. We report raw
p-values for 9 comparisons (4 McNemar tests, 5 permutation tests) and read them with that in mind.

## 4. Experiments

All experiments ran on a free Kaggle T4 GPU (16 GB) in three unattended sessions;
CPU numbers come from the same Kaggle machine. Unless said otherwise, results are on
our SciFact test split (300 claims), brackets are 95% bootstrap intervals, and
"± x" is the standard deviation over the 3 training seeds.

### 4.1 Retrieval (A1, A2)

Table 1 – Retrieval on the 188 test claims that have gold evidence.

| Retriever | R@1 | R@5 | R@20 | MRR |
| --- | --- | --- | --- | --- |
| BM25 (ours) | 0.712 | 0.859 | 0.943 | 0.798 |
| BM25 + rerank | 0.750 | 0.913 | 0.945 | 0.840 |
| Dense (BGE-small) | 0.744 | 0.898 | **0.979** | 0.834 |
| Dense + rerank | **0.760** | **0.935** | 0.975 | **0.856** |
| Hybrid (RRF) | 0.741 | 0.912 | 0.966 | 0.836 |
| Hybrid + rerank | 0.755 | 0.934 | 0.972 | 0.850 |

Our BM25 is already a strong baseline (R@5 = 0.859), which is expected on SciFact
where claims reuse the technical words of the abstract. The dense encoder finds more
of the gold abstracts in the top 20 (0.979), and the cross-encoder turns this into a
better top 5: dense + rerank gains +0.076 R@5 over BM25 (paired permutation test,
p = 0.0007). Hybrid fusion alone is close to dense alone (+0.052 vs +0.039 over BM25),
and once the reranker is applied dense and hybrid reach nearly the same R@5 (0.935 vs
0.934). The reranker adds more than the choice between dense and hybrid.

### 4.2 Hyperparameter search

Table 2 – Dev macro-F1 of the verifier (gold evidence), batch size 16, one seed.

| Learning rate | 3 epochs | 5 epochs |
| --- | --- | --- |
| 1e-5 | 0.824 | 0.842 |
| 2e-5 | 0.829 | 0.855 |
| 5e-5 | **0.858** | 0.855 |

Batch size 32 was planned but did not fit in the T4 memory in float32. All six runs
are within 0.034 of each other, and the best one (5e-5, 3 epochs) is only 0.003 above
our default (2e-5, 5 epochs), a small difference in a single-seed search. We kept the
default and evaluated it across three seeds.

### 4.3 Main results and baselines

Table 3 – End-to-end results on SciFact test (300 claims).

| System | Accuracy | Macro-F1 | Time / claim |
| --- | --- | --- | --- |
| Majority class | 0.413 [0.360, 0.467] | 0.195 [0.176, 0.212] | – |
| Claim only (TF-IDF + LR) | 0.437 [0.377, 0.497] | 0.401 [0.345, 0.457] | – |
| Rules (numbers, negation, overlap) | 0.517 [0.460, 0.577] | 0.475 [0.422, 0.533] | – |
| BM25 + zero-shot NLI | 0.580 [0.527, 0.637] | 0.570 [0.515, 0.625] | – |
| Full retrieval + zero-shot NLI | 0.617 [0.563, 0.677] | 0.609 [0.550, 0.667] | – |
| Full retrieval + LLM judge (Qwen2.5-7B, 4-bit) | 0.687 [0.633, 0.733] | 0.671 [0.612, 0.722] | 15.4 s (T4) |
| **ClaimGuard, 3 seeds** | 0.701 ± 0.012 | **0.693 ± 0.010** | 0.6 s (T4), 1.0 s (CPU) |
| ClaimGuard, main system (seed 13, T = 2.42) | 0.683 [0.633, 0.733] | 0.676 [0.620, 0.726] | 0.6 s (T4) |

Per class (main system): SUPPORTED F1 0.742, REFUTED 0.667, NEI 0.618.

The claim-only baseline reaches 0.401 macro-F1 without reading any evidence, twice the
majority class: part of the label can be guessed from the wording of the claim alone,
as already reported for FEVER (Schuster et al., 2019). This baseline gives us a reference for how well a system can do from wording alone. Reading evidence with a
zero-shot NLI model adds 0.21, and fine-tuning on SciFact adds another 0.08. The
fine-tuned 184M-parameter verifier scores slightly above the 7B LLM judge in this comparison (0.693 vs 0.671; the confidence intervals overlap and we did not test the difference), while being about 26 times faster on the same GPU and able to run
on CPU. NEI is the hardest class: it is the one where "the passage talks about the
topic" must not be read as "the passage supports the claim" (Section 5).

### 4.4 Ablations (A3–A6)

Table 4 – Ablations on the main system (macro-F1 on test).

| Ablation | Setting | Macro-F1 |
| --- | --- | --- |
| A2 reranker | with / without | 0.676 / 0.676 |
| A3 evidence passages k | 1 / 3 / 5 / 10 | 0.678 / 0.676 / 0.675 / 0.667 |
| A4 aggregation | majority / max-confidence / noisy-OR | 0.369 / 0.681 / 0.676 |
| A5 verifier | zero-shot NLI / plain DeBERTa + SciFact / NLI + SciFact (3 seeds) | 0.609 / 0.509 / 0.693 |
| A6 evidence | retrieved / gold (oracle) | 0.676 / 0.743 |

**A2.** The reranker improves retrieval (Table 1). Individual verdicts change in both directions:
McNemar b = 20, c = 21, p = 1.0. Moving a relevant abstract into the top 5 can improve retrieval even if the selected sentences still leave the decisive detail out. The paired counts show that gains and losses nearly cancel.
**A3.** k barely matters between 1 and 5 (Figure `ablation_k.png`); k = 10 loses a
little because more neutral passages enter the aggregation. k = 3 was chosen on dev.
**A4.** This is the largest effect of the study. The majority baseline scores
0.369, because several neutral passages can outweigh a passage that clearly supports or refutes the claim. Noisy-OR pools evidence for support and refutation across passages, so a single decisive passage can carry substantial weight (noisy-OR vs majority: b = 106, c = 37, p < 0.0001; noisy-OR vs
max-confidence: p = 1.0).
**A5.** Starting from an NLI checkpoint is essential: the same architecture
fine-tuned from plain DeBERTa-v3-base gets 0.509, below the zero-shot NLI model.
With only 687 SciFact training claims, the verifier benefits from prior training on entailment and contradiction.
**A6.** With gold evidence, 24 of the 95 errors are fixed but 5 new ones appear, a
net gain of 19 (McNemar p = 0.0005), and macro-F1 rises to 0.743. Providing gold evidence helps, but most original errors persist. Better retrieval alone would leave many verdicts unresolved.

### 4.5 Calibration and abstention

The raw model is over-confident: before scaling, ECE is 0.260. Temperature scaling
with T = 2.42 (fitted on dev, at the claim level) reduces it by 46% to 0.141
(Figure `reliability.png`). The area under the risk–coverage curve (AURC) rises from 0.142 to 0.153. Lower is better: this measure tracks errors among the claims retained as the confidence threshold varies. Scaling before noisy-OR can change the confidence ranking even as calibration improves.

Table 5 – Abstention after temperature scaling. An alert is a REFUTED or NEI verdict.

| Threshold | Coverage | Selective accuracy | Alert precision | Alert recall |
| --- | --- | --- | --- | --- |
| 0.0 | 1.000 | 0.683 | 0.841 | 0.750 |
| 0.5 (chosen on dev) | 0.980 | 0.694 | 0.850 | 0.739 |
| 0.7 | 0.823 | 0.753 | 0.876 | 0.642 |
| 0.8 | 0.697 | 0.794 | 0.905 | 0.540 |
| 0.9 | 0.363 | 0.881 | 0.935 | 0.165 |

Macro-F1 in Tables 3–4 is always computed on the model's best guess; UNCERTAIN
answers only appear here. The threshold of 0.5 chosen on dev answers 98% of the
claims. A user who wants alerts that are right 9 times out of 10 can move to 0.8 and
keep 70% coverage, at the cost of missing about half of the problems (alert recall
0.540). Figure `risk_coverage.png` shows the whole trade-off.

### 4.6 Stress tests

Table 6 – Contrast set (125 edited claims generated from the 124 SUPPORTED test
claims, gold evidence given; a rule only applies when the claim has a number, a
negation or a direction word). "Detected" = the edited claim is no longer predicted SUPPORTED.

| Edit | n | Fine-tuned | Zero-shot NLI | Rules |
| --- | --- | --- | --- | --- |
| Number changed | 20 | **0.30** | 0.70 | 1.00 |
| Negation added / removed | 64 | **0.97** | 0.92 | 0.94 |
| Direction word swapped | 41 | **0.88** | 0.98 | 0.63 |
| Originals predicted SUPPORTED | | **0.87** | 0.47 | 0.35 |
| Consistency (original right and edit detected) | | **0.728** | 0.424 | 0.248 |

The fine-tuned model most often gets both parts of a pair right: it accepts the original claim and flags the edit. The zero-shot model and the
rules look good on "detected" only because they reject most claims anyway (they keep
fewer than half of the unedited originals). The clear weakness is numbers: the
fine-tuned verifier misses 70% of changed numbers, so "a rate of 4.35%" and "a rate
of 3.45%" look almost identical to it. Fine-tuning made this worse than the
zero-shot model (0.30 vs 0.70): one possible explanation is that the SciFact training examples provide limited practice with changes to a single value.

Table 7 – Verifier given the evidence, other test sets.

| Test set | Metric | Fine-tuned | Zero-shot NLI | Rules |
| --- | --- | --- | --- | --- |
| Climate-FEVER (1,381) | macro-F1 | **0.522** [0.495, 0.547] | 0.405 [0.381, 0.431] | 0.386 [0.361, 0.410] |
| Climate-FEVER | ECE | 0.113 | 0.434 | 0.063 |
| French banking (60) | macro-F1 | **0.949** [0.885, 1.000] | 0.934 [0.866, 0.985] (multilingual) | 0.765 [0.653, 0.868] |

On Climate-FEVER, a new domain with noisier evidence, the fine-tuned verifier keeps a
clear lead over the zero-shot model (+0.12) and stays much better calibrated (ECE
0.113 vs 0.434). On our French banking set, the English verifier gets 57 of 60 right,
slightly above the multilingual NLI model (56 of 60). Its 3 errors are all
"missing information" claims, the hardest category for both models (17 of 20). The set is small and we wrote it
ourselves, so this only shows that the pipeline is usable on French text, not that it
is better than the multilingual model.

### 4.7 Sentences written by LLMs

Table 8 – LLM-AggreFact, balanced accuracy, 100 claims per sub-dataset.

| Sub-dataset | Fine-tuned | Zero-shot NLI | Rules |
| --- | --- | --- | --- |
| AggreFact-CNN | 0.583 | **0.739** | 0.522 |
| AggreFact-XSum | 0.603 | 0.666 | **0.677** |
| TofuEval-MediaS | 0.638 | **0.686** | 0.610 |
| TofuEval-MeetB | **0.796** | 0.795 | 0.762 |
| Wice | 0.659 | **0.712** | 0.512 |
| ExpertQA | 0.574 | 0.539 | **0.586** |
| ClaimVerify | **0.732** | 0.695 | 0.657 |
| FactCheck-GPT | 0.733 | **0.740** | 0.584 |
| Lfqa | **0.860** | 0.800 | 0.716 |
| RAGTruth | **0.769** | 0.748 | 0.680 |
| Reveal | **0.850** | 0.812 | 0.664 |
| **Mean** | 0.709 | **0.721** | 0.634 |

On average, fine-tuning on SciFact does not help on LLM outputs (0.709 vs 0.721). The
picture depends on the kind of text: the fine-tuned model is better on the RAG and
long-form QA sets (Lfqa, RAGTruth, Reveal, ClaimVerify), where a sentence is checked
against retrieved passages as in SciFact, and worse on summaries of news and of
dialogues (AggreFact-CNN −0.16, TofuEval-MediaS −0.05). With 100 claims per
sub-dataset, small differences should be interpreted cautiously.

### 4.8 Efficiency

Table 9 – Verifier on CPU (one claim = 5 passages, 30 runs), gold-evidence accuracy
on 209 pairs.

| Version | Size | p50 | p95 | Accuracy |
| --- | --- | --- | --- | --- |
| FP32 (184M parameters) | 738 MB | 1,037 ms | 1,103 ms | **0.856** |
| INT8 dynamic, all linear layers | 481 MB | 918 ms | 954 ms | 0.407 |
| INT8 dynamic, feed-forward only | 568 MB | 949 ms | 996 ms | 0.455 |

On GPU the full pipeline takes 627 ms per claim (p50; p95 790 ms). Dynamic INT8
quantization saves 35% of the size but only 11% of median CPU inference time. Accuracy drops sharply,
even when the attention layers are kept in FP32. We kept FP32: at about one second
per claim on CPU, the demo is usable without a GPU.

## 5. Error Analysis

**Protocol.** We export every wrong prediction of the main system on the SciFact test
set (95 errors) and add the 15 errors of a random sample of Climate-FEVER, which
gives 110 errors. Each error is first tagged automatically: *retrieval miss* (the
claim has gold evidence and none of it reached the verifier) or *verification
error*, presence of a number, presence of a negation. Then each error receives one
category (taxonomy in Table 10), with a short note justifying it. The annotation was
done with the help of an AI assistant; the categories and notes were then reviewed.
To check the consistency of the categories, 25 errors drawn at random were annotated a second time by the author alone, without AI assistance and without access to the first labels, the same day a few hours after the first pass. The two passes agree on 24 of the 25 errors (96%), Cohen's κ = 0.95 (`results/kappa.json`). The only change is claim 1280, moved from domain vocabulary to related but insufficient, two categories that overlap when a technical term hides a missing link. The short delay probably inflates the agreement, so this number shows that the categories are usable, not that they are fully reliable.

### 5.1 Quantitative breakdown

Table 10 – Error categories (110 errors).

| Category | Definition | SciFact | Climate | Total | Main confusion |
| --- | --- | --- | --- | --- | --- |
| Wrong sentences | right document, but the sentences given to the verifier are not the ones that decide | 25 | 7 | 32 (29%) | SUPPORTED → NEI |
| Related but insufficient | the model predicts SUPPORTED from related evidence that fails to establish an essential part of the claim | 17 | 2 | 19 (17%) | NEI → SUPPORTED |
| REFUTED vs NEI | evidence goes against the claim without contradicting it, or the reverse | 11 | 4 | 15 (14%) | NEI → REFUTED |
| Domain vocabulary | the decision depends on a technical term or relation the model misreads | 15 | 0 | 15 (14%) | NEI → SUPPORTED |
| Evidence not retrieved | no gold document reached the verifier | 9 | 0 | 9 (8%) | REFUTED → NEI |
| Numbers | wrong reading of a number, unit or statistical result | 4 | 1 | 5 (5%) | SUPPORTED → NEI |
| Multi-sentence evidence | the proof needs two sentences together | 5 | 0 | 5 (5%) | SUPPORTED → NEI |
| Negation / quantifier | wrong scope of "not", "no evidence", "often" | 4 | 1 | 5 (5%) | REFUTED → SUPPORTED |
| Gold label questionable | our reading of the evidence disagrees with the gold label | 5 | 0 | 5 (5%) | NEI → SUPPORTED |

Figure `errors.png` shows the same counts. Three numbers stand out.

* **Errors are made with confidence.** 44 of the 110 errors have a confidence of at
  least 0.8 (mean 0.728 over the 110, 0.746 over the 95 SciFact errors). A threshold
  set at that level would still let these errors through.
* **The most frequent confusion is NEI → SUPPORTED** (39 of 110). On SciFact, NEI
  claims have the highest error rate (48 of 112, 43%), against 34% for REFUTED (22 of
  64) and 20% for SUPPORTED (25 of 124).
* **Numbers and negations are not where SciFact errors are.** Claims with a number
  have an error rate of 29% (33 of 114) against 33% for the others (62 of 186); claims
  with a negation 24% (7 of 29) against 33%. This looks like it contradicts the
  contrast set, but it does not: the number flag groups project names ("1,000 Genomes project") with genuine prevalence claims ("1/2000 in the UK"). The contrast set isolates a different question: does the verdict change when only the value changes?

### 5.2 Qualitative analysis

(Evidence shortened. Translated from our annotation notes.)

**Wrong sentences.** Claim 128, *"Arterioles have a larger lumen diameter than
venules"* (gold NEI, predicted SUPPORTED, 0.88). The retrieved abstract compares the
aorta with the coronary arteries; venules never appear. The sentences given to the
verifier speak about vessel diameter, so the model accepts the comparison. Claim 3
(gold SUPPORTED, predicted NEI, 0.83) is the reverse case: the claim is supported
by its gold evidence, but the sentences given to the verifier come from a study of
the gene PPM1D and say nothing about the penetrance of the 1,000 Genomes variants. In both cases the error comes from the sentence selection step, which ranks
sentences by similarity to the claim instead of by how much they decide it.

**Related but insufficient.** Claim 36, *"A deficiency of vitamin B12 increases blood
levels of homocysteine"* (gold NEI, predicted SUPPORTED, 0.74). The abstract reports a
diet that lowers homocysteine while vitamin B12 *stays unchanged*: same topic, same
two entities, but nothing about a deficiency. Claim 350 (0.82) is similar: IF3 takes
part in translation initiation, which does not show that it discriminates between the
two tRNAs. The model checks that the claim's entities and relation appear in the
passage, not that the specific link the claim asserts is established.

**REFUTED vs NEI.** Claim 213, *"CRP is not predictive of postoperative mortality
following CABG surgery"* (gold NEI, predicted REFUTED, 0.92). The abstract links CRP
to long-term mortality after bypass surgery; the claim leaves the postoperative time horizon unspecified. An association over long-term follow-up does not by itself settle the prediction question as worded.

**Domain vocabulary.** The pair of claims 268 and 269 is telling: *"Cold exposure
increases BAT recruitment"* is predicted SUPPORTED (0.82) and *"Cold exposure reduces
BAT recruitment"* is predicted REFUTED (0.82), while both are NEI (the second one is
counted under REFUTED vs NEI). The abstract says
that cold makes brown adipose tissue express Ucp1, a thermogenic response, but never
measures recruitment of the tissue. The model treats "express Ucp1" and "recruit BAT"
as interchangeable, although the passage measures only the former.

**Negation scope.** Claim 1290, *"There is an inverse relationship between hip
fractures and statin use"* (gold SUPPORTED, predicted REFUTED, 0.91). The abstract
says the risk falls with statin use and that there is "no evidence of such
relationships with nonstatin lipid-lowering agents". The model attaches "no
evidence" to statins instead of to the other drugs.

### 5.3 What the errors tell us

1. **The verifier judges topic overlap more than support.** Related but insufficient,
   domain vocabulary and most NEI → SUPPORTED errors (39 of 110) share the same
   pattern. *Fix:* add hard negatives built from the gold abstract itself (sentences of
   the right abstract that do not support the claim, labelled NEI) so that the model
   learns that the right topic is not enough.
2. **Wrong sentence selection is the most frequent annotated category** (32 errors), while reranking documents left the end-to-end score unchanged in this experiment. *Fix:* select sentences with the verifier's own scores, or give the
   verifier the whole abstract when it fits in 512 tokens.
3. **Confidence does not separate these errors.** 40% of them are made with at least
   0.8 confidence, so abstention protects against hesitant errors only. *Fix:* report
   the decisive passage to the user, as the demo does, so a human can check it in a
   few seconds.
4. **Numbers are a hidden risk.** They are rare in SciFact errors but the contrast
   set shows that the model misses 70% of changed numbers, which matters most for the
   use cases we target (banking, prices, dates). *Fix:* a rule that compares the
   numbers of the claim and of the decisive sentence before the verifier, as in our
   lexical baseline, which flags 100% of number edits (it also rejects many true
   claims, so it should only send a warning, not decide alone).
5. **Some labels are debatable** (5 of 110). On a test set of 300 claims, a few disputed decisions warrant a closer look when comparing nearby scores in Table 3.

## 6. Discussion

[[TODO after results: 2-3 paragraphs. What worked and why; what did not; how far the
SciFact conclusions transfer (Climate-FEVER, French set); cost / quality trade-off
against the LLM judge]]

**Limitations.** (1) Faithfulness, not truth: a claim that repeats a wrong document
is SUPPORTED. (2) The test set has 300 claims, so intervals are wide and small
differences are not significant. (3) SciFact annotations are incomplete, so some hard
negatives used in training and some "false" positives at test time may be real
evidence. (4) English training data only; the French set is small and synthetic.
(5) Sentences are checked one by one without resolving pronouns, and claims that need
two documents together are not handled. (6) The contrast set labels come from rules. (7) The error categories were first assigned with AI assistance, and the consistency check is a same-day re-annotation by one person, not an agreement between two independent annotators.
(8) Long documents with tables are out of scope.

## 7. Conclusion and Future Work

[[TODO 1 paragraph with the 3 main findings]]

Future work: a learned aggregator when more data is available; claim decomposition
with a small LLM; multi-hop claims (HoVer); calibration per claim type, since a
confidence on an amount does not carry the same risk as one on a description; a
the full LLM-AggreFact benchmark and a direct comparison with MiniCheck; a hook that runs the verifier
after every assistant answer.

## References

1. Cormack, G., Clarke, C., Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
2. Diggelmann, T., Boyd-Graber, J., Bulian, J., Ciaramita, M., Leippold, M. (2020). CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims. arXiv:2012.00614.
3. Gardner, M., et al. (2020). Evaluating Models' Local Decision Boundaries via Contrast Sets. Findings of EMNLP.
4. Geifman, Y., El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. NeurIPS.
5. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. (2017). On Calibration of Modern Neural Networks. ICML.
6. He, P., Gao, J., Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. ICLR.
7. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP.
8. Laban, P., Schnabel, T., Bennett, P., Hearst, M. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization. TACL.
9. Laurer, M., van Atteveldt, W., Casas, A., Welbers, K. (2024). Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI. Political Analysis.
10. Min, S., et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP.
11. Nogueira, R., Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
12. Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.
13. Schuster, T., et al. (2019). Towards Debiasing Fact Verification Models. EMNLP.
14. Tang, L., Laban, P., Durrett, G. (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. EMNLP.
15. Thorne, J., Vlachos, A., Christodoulopoulos, C., Mittal, A. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL.
16. Wadden, D., et al. (2020). Fact or Fiction: Verifying Scientific Claims. EMNLP.
17. Wadden, D., et al. (2022). MultiVerS: Improving Scientific Claim Verification with Weak Supervision and Full-Document Context. Findings of NAACL.
18. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding. arXiv:2309.07597.

<!-- Before submitting: open each reference on Google Scholar and check authors, year and venue. -->
