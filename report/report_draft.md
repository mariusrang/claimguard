---
title: "ClaimGuard: Evidence-Based Claim Verification with Calibrated Abstention"
author: "Marius Rang — NLP final project, topic 2 (Fact Verification with Evidence Retrieval)"
geometry: margin=2.2cm
fontsize: 10.5pt
---

> **Draft status.** Sections 1–3 and the experimental protocol are written. Every
> `[[TODO ...]]` is a number, table, figure or paragraph that can only be written
> after the runs (`notebooks/01_colab_run_all.ipynb`) and the manual annotation.
> Nothing in this file is a result yet. Remove this box before exporting to PDF.

## Abstract

Language models produce fluent answers that are not always supported by their
sources. We present ClaimGuard, a pipeline that verifies a claim against a document
collection and returns SUPPORTED, REFUTED or NOT ENOUGH INFO with the evidence and a
calibrated confidence. The pipeline combines hybrid retrieval (BM25 and a dense
encoder fused with Reciprocal Rank Fusion), cross-encoder reranking, a DeBERTa-v3
verifier fine-tuned on SciFact, multi-evidence aggregation and temperature scaling.
On the SciFact test set the full system reaches [[TODO macro-F1 with 95% CI]]
against [[TODO]] for a zero-shot NLI baseline and [[TODO]] for an open-source LLM
judge. An oracle experiment shows that [[TODO x%]] of the errors come from retrieval.
On sentences written by LLMs (LLM-AggreFact) the same pipeline reaches a balanced
accuracy of [[TODO]]. After temperature scaling the expected calibration error drops from [[TODO]] to
[[TODO]], and with an abstention threshold of [[TODO]] the precision of the alerts
reaches [[TODO]] for a coverage of [[TODO]]. We stress-test the verifier on a
contrast set with edited numbers, negations and antonyms, on a French banking test
set and on Climate-FEVER, and analyse [[TODO n ≥ 100]] errors by hand. The system
ships with a demo, a Docker image and an MCP server with an audit log.

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
by LLMs (LLM-AggreFact); (vi) a manual analysis of [[TODO n]] errors; and (vii) a
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
the edit). [[TODO: n examples per rule; share of a hand-checked sample of 50 where
the edit really breaks the support]]

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
the benchmark's own metric. Because of the Colab budget we use 100 claims per
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
p-values for [[TODO n]] comparisons and read them with that in mind.

## 4. Experiments

### 4.1 Retrieval (A1, A2)

[[TODO table from results/tables.md: BM25 / dense / hybrid, with and without
reranker: R@1, R@5, R@20, P@5, MRR; permutation tests against BM25]]

[[TODO 3-4 sentences: which retriever wins, is the gain significant, what the
reranker adds, what recall@50 says about the ceiling for the next stages]]

### 4.2 Hyperparameter search

All experiments run on a free Colab T4 GPU. [[TODO table from results/sweep.json
(12 runs: learning rate × batch size × epochs), the chosen configuration, one
sentence on how sensitive the model is to the learning rate]]

### 4.3 Main results and baselines

[[TODO table: 5 baselines + our system (mean ± std over 3 seeds): accuracy, macro-F1
with CI, FEVER score, evidence F1; McNemar p-values against the main system]]

[[TODO paragraph: what the claim-only baseline says about the dataset; gap between
zero-shot NLI and the fine-tuned model; position of the LLM judge, with its latency]]

### 4.4 Ablations (A3–A6)

[[TODO table A3: k = 1, 3, 5, 10 + figure ablation_k.png]]
[[TODO table A4: majority / max-confidence / noisy-OR + McNemar]]
[[TODO table A5: zero-shot NLI / plain encoder + SciFact / NLI + SciFact]]
[[TODO table A6: retrieved vs gold evidence; share of errors fixed by gold evidence]]

[[TODO one short paragraph per ablation with the conclusion]]

### 4.5 Calibration and abstention

[[TODO figure reliability.png, figure risk_coverage.png; T, ECE and AURC before and
after; table: threshold, coverage, selective accuracy, alert precision, alert recall]]

[[TODO paragraph: is the raw model over-confident; what the chosen threshold costs in
coverage; how many high-confidence errors remain (these cannot be caught by abstention)]]

### 4.6 Stress tests

[[TODO table contrast set: not-SUPPORTED rate per edit type and consistency, for the
rules, the zero-shot NLI model and the fine-tuned verifier]]
[[TODO table French set: English verifier / multilingual verifier / rules, accuracy
per category]]
[[TODO table Climate-FEVER: accuracy, macro-F1, ECE (calibration under domain shift)]]

Measured so far, without any neural model (CPU runs, 23 Nov): the rule-based verifier
reaches 0.77 accuracy on the French set (n = 60) and 0.49 accuracy / 0.39 macro-F1 on
Climate-FEVER (n = 1,381). On the contrast set it stops saying SUPPORTED for 100% of the
number edits and 94% of the negation edits, but it calls only 35% of the untouched
claims SUPPORTED, so its consistency is 0.25: a high flip rate alone is misleading.

### 4.7 Sentences written by LLMs

[[TODO table from results/tables.md: balanced accuracy per LLM-AggreFact sub-dataset
and mean, for the rules, zero-shot NLI and the fine-tuned verifier; latency per claim]]

[[TODO paragraph: does fine-tuning on SciFact help or hurt on LLM outputs; which
sub-datasets are hardest and why (long documents? RAG vs summaries?)]]

### 4.8 Efficiency

[[TODO table: FP32 vs INT8 on CPU: size, p50 and p95 latency per claim, seconds per
1,000 claims, macro-F1 with gold evidence; end-to-end latency on GPU; LLM judge
latency for comparison]]

## 5. Error Analysis

<!-- at least 2 pages in the final PDF -->

**Protocol.** We export every wrong prediction of the main system on the test set
([[TODO n]] errors), completed with [[TODO n]] errors sampled from Climate-FEVER to
reach [[TODO ≥ 100]]. Each error is first tagged automatically: *retrieval miss* (the
claim has gold evidence and none of it was given to the verifier) or *verification
error*; presence of a number; presence of a negation. Then each error receives one
category by hand (taxonomy in Table [[TODO]]). To check the consistency of the
categories, 25 errors were annotated a second time one week later:
agreement [[TODO]], Cohen's κ = [[TODO]].

### 5.1 Quantitative breakdown

[[TODO figure errors.png + table: category, count, share, share of retrieval misses,
mean confidence]]
[[TODO table: confusion matrix of the errors (gold × predicted)]]
[[TODO error rate for claims with / without a number, with / without a negation, by
claim length; error rate by confidence bin]]

### 5.2 Qualitative analysis

[[TODO for each of the 4-5 biggest categories: 2 examples with the claim, the
evidence shown to the model, gold and predicted labels, and one sentence explaining
what went wrong]]

### 5.3 What the errors tell us

[[TODO 4-5 findings, each ending with what to do about it. Template:
"x% of the errors are retrieval misses → improving the verifier cannot fix them;
the next step is ...". "High-confidence errors are mostly of type ... → abstention
does not help there; ...". "Gold labels look questionable in n cases → ..."]]

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
two documents together are not handled. (6) The contrast set labels come from rules.
(7) Long documents with tables are out of scope.

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
