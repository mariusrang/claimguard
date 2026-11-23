# Design decisions (and why)

I keep this file to remember why each choice was made, and to be able to defend it.
A coding assistant helped me write code faster, but every choice below is one I have
to be able to explain without notes.

## Scope

- **Three ideas instead of seven bricks.** After the expert review I cut the learned
  aggregator, the bi-encoder fine-tuning, DeBERTa-large, the fine-tuned decomposition
  model and the hosted Space. Reason: one person, limited GPU time, and a test set of
  about 300 claims where small gains cannot be told apart from noise anyway.
- **Not "small model beats big LLM".** MiniCheck already showed it. The LLM judge is
  one baseline. What I add is calibration, abstention, numeric stress tests and a
  deployment with an audit trail.

## Data

- **Official dev = my test.** SciFact test labels are not public. So I never touch the
  official dev set for any decision; model selection, temperature and threshold are
  all chosen on my own dev split (15% of train, stratified, seed 13).
- **Claim label.** SUPPORTED / REFUTED if the claim has gold evidence with that label,
  NEI if it has no evidence. This is the 3-way claim-level task; the official SciFact
  metric is at abstract level, so my numbers are not directly comparable to the
  leaderboard. I say it in the report.
- **Climate-FEVER: DISPUTED dropped.** These claims have both supporting and refuting
  evidence, which does not fit three labels.
- **French set: synthetic and small.** 20 invented bank documents, 3 claims each. It
  is a stress test, not a benchmark. The REFUTED claims change one number, date or
  negation because this is the failure a bank cares about. Limits are in its data card.

## Retrieval

- **BM25 written by hand.** About 60 lines with an inverted index. I wanted to know
  the formula, and `tests/test_retrieval.py` checks a score against a hand computation.
- **No FAISS.** 5k abstracts × 384 dimensions is a 2M-float matrix; one matrix product
  is exact and takes milliseconds. FAISS would add a dependency for no gain here.
- **RRF instead of mixing scores.** BM25 scores and cosine similarities are on
  different scales. RRF only uses ranks, so there is nothing to normalise or tune
  (k = 60 is the value of the original paper).
- **Dense model used as is.** Fine-tuning a bi-encoder on about 700 claims is unlikely to
  beat a recent embedding model, and it would cost GPU time. Measured in A1, not assumed.
- **Query prefix for BGE.** BGE v1.5 expects an instruction before short queries.

## Reranking and sentence selection

- **Cross-encoder on the top-50 only.** It is accurate because it reads claim and
  passage together, and too slow for the whole corpus.
- **Same cross-encoder picks the sentences.** One model, two jobs: rank abstracts,
  then rank the sentences inside each kept abstract (3 per abstract, in reading order).
  A dedicated rationale selector trained on SciFact would be better; future work.

## Verifier

- **Start from an NLI checkpoint.** Verification is NLI with other names
  (entailment / contradiction / neutral), and SciFact is small, so NLI pre-training
  should help. Ablation A5 measures it against a plain DeBERTa encoder.
- **Label mapping by name, never by index.** Checkpoints order their classes
  differently. `map_model_labels` reads `id2label` and fails loudly if it cannot find
  the three classes. There is a test with three different orders.
- **Evidence first, claim second, truncate the evidence only.** NLI convention
  (premise, hypothesis). If the pair is too long, the claim must stay complete.
- **Hard negatives.** Without them the model only sees NEI on unrelated text and
  learns "same topic = supported". A BM25-retrieved, non-gold abstract is on topic but
  is not evidence. Risk: SciFact annotations are incomplete, so some negatives may
  actually be evidence. Accepted, and mentioned as a limit.
- **Own training loop.** AdamW, no weight decay on biases and LayerNorm, linear warmup
  (10%) then linear decay, gradient clipping at 1.0, best epoch on dev macro-F1.
- **fp16 off by default.** DeBERTa-v3 is known to be unstable in half precision
  sometimes; SciFact is small enough to train in full precision on a T4.
- **Macro-F1 for model selection.** Classes are not balanced; accuracy would hide a
  bad REFUTED class.

## Aggregation

- **Three rule-based strategies, noisy-OR by default.** Most retrieved documents are
  irrelevant, so a majority vote is dominated by NEI. Noisy-OR says "supported if at
  least one document supports". The default is my prior; A4 decides with a McNemar test.
- **No learned aggregator.** About 700 training claims; it would overfit and I could
  not conclude anything.

## Calibration and abstention

- **Temperature scaling.** One parameter, fitted on dev by minimising the NLL (1-D
  bounded search on log T). It cannot change the predicted label, only the confidence.
- **Temperature is fitted on the claim-level probabilities.** My first version fitted T
  on the passage pairs. On the real SciFact data (lexical baseline) that made the
  claim-level ECE worse on test (0.206 -> 0.258), because the aggregation changes the
  probabilities. Now T divides the passage logits but is chosen to minimise the NLL of
  the final claim probabilities on dev, i.e. of the confidence the user actually sees.
- **k, the aggregation and T are chosen together on dev.** For each of the 12 (k,
  aggregation) settings, T is fitted, then the setting with the best dev macro-F1 wins.
  Baselines go through the same dev selection (with T = 1), so the comparison is fair.
- **Majority = soft vote (mean of the probabilities).** With a hard vote and k = 1 the
  confidence is always 100%, which makes abstention impossible.
- **Threshold chosen on dev for a target alert precision (0.9).** Then reported on
  test. Choosing it on test would be cheating.
- **Headline metric: alert precision.** If the tool flags too many correct sentences,
  people stop reading the alerts. This is the number I would give to a non-technical
  person.

## Evaluation

- **FEVER-style score adapted to abstracts.** Right label and at least one gold
  evidence abstract among the evidence given to the verifier (NEI claims: label only).
- **Bootstrap CI everywhere, McNemar between systems, permutation test for retrieval.**
  With 300 test claims a 2-point difference is often noise, and I prefer to show it.
- **No multiple-comparison correction.** I report the raw p-values and say that there
  are several comparisons, instead of adding Holm-Bonferroni on a handful of tests.
- **Oracle experiment.** Gold evidence instead of retrieved evidence. The share of
  errors that disappears is the share caused by retrieval.

## Stress tests

- **Contrast set by rules.** One edit per example: a number, a negation or an antonym.
  Labels are noisy (an edited claim can still be true by accident), so the metric is
  the rate of predictions that stop being SUPPORTED, and I check a sample by hand.
- **Consistency.** Right on the original pair AND reacts to the edit. A model that
  always says REFUTED would have a perfect flip rate; consistency catches that.

## Checking LLM answers (extension)

- **LLM-AggreFact, subsampled.** It is the reference benchmark for "is this LLM sentence
  supported by its document", with human labels. 100 claims per sub-dataset keeps one
  Colab session reasonable. I report it as a subsample, not as a leaderboard number.
- **Sources mode, not the whole document.** Documents are often longer than 384 tokens.
  Chunking + BM25 + reranker reuses the main pipeline, so the extension also tests it.
- **Binary mapping.** The benchmark has supported / not supported, so REFUTED and NEI
  both count as "not supported". Balanced accuracy, because some sub-datasets are very
  unbalanced.
- **Decomposition = sentence splitting.** Simple and predictable. Its limit (pronouns
  not resolved) is written in the report.

## Compute

- **Colab only.** No school GPU (professor, 21 November). DeBERTa-v3-base, the LLM judge
  in 4-bit (Qwen2.5-7B, fallback 3B), results backed up to Google Drive after each step.
  SciFact only gives about 1,570 training pairs of about 85 words, so one fine-tuning run
  takes about 3 minutes: the full 12-run sweep is affordable.
- **CPU runs first.** Everything that needs no neural model (BM25 retrieval, the
  no-evidence baselines, the rule-based pipeline, the stress tests with rules) was run
  on CPU on the real data before using any GPU time. It found two problems early (see
  calibration and aggregation above).

## Deployment

- **Offline mode.** BM25 + a rule-based verifier, no download. It is my lexical
  baseline, and it lets the tests, the CI, Docker and the MCP server run anywhere.
- **Dynamic INT8 quantization (PyTorch) instead of ONNX.** Fewer moving parts, and I
  can test it in CI on a tiny model. I measure size, latency and accuracy before and
  after instead of assuming it is free.
- **Audit log as JSONL.** Append-only, one verdict per line with the model version
  (verifier name, aggregation, temperature, threshold). Easy to export and to diff.
- **MCP: four tools, short outputs.** The assistant gets the decision, the confidence
  and the best evidence; the full record stays in the audit log.
- **Sources mode chunks documents (6 sentences) and uses BM25 on the fly.** Simple and
  it works on long documents, but tables are not handled.

## Things I should be able to answer in the video

1. Why can temperature scaling not change accuracy?
2. Why does RRF not need score normalisation?
3. What does the claim-only baseline tell about the dataset?
4. Why are hard negatives needed, and what is their risk on SciFact?
5. What is the difference between ECE and AURC?
6. Why is a perfect flip rate on the contrast set not enough?
7. What exactly does "SUPPORTED" guarantee, and what does it not?
