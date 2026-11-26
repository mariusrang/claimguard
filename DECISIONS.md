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

- **Official dev = my test.** SciFact test labels are not public. So the official dev
  set is my test set; model selection, temperature and threshold are all chosen on my
  own dev split (15% of train, stratified, seed 13). One exception, told in
  "Calibration" below: a look at the test ECE of the rule-based baseline made me change
  how T is fitted. Nothing else was decided from test numbers.
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
  bounded search on log T). On one passage it cannot change the predicted label, only
  the confidence. After noisy-OR over several passages it can: T changes how strongly
  each passage counts, which is why seed 13 gets 0.685 macro-F1 with T = 1 and 0.676
  with T = 2.42.
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
- **Oracle experiment, on the claims with evidence only.** My first version counted all
  300 claims, but the 112 NEI claims have no gold evidence, so their 48 errors could not
  change by construction. Session 5 compares on the 188 claims with evidence, in two
  steps: gold abstracts with my own sentence selection (tests retrieval), then gold
  sentences (tests sentence selection). Result: retrieval is not the problem (0.739 vs
  0.750), sentence selection is (0.851).

## Stress tests

- **Contrast set by rules.** One edit per example: a number, a negation or an antonym.
  Labels are noisy (an edited claim can still be true by accident), so the metric is
  the rate of predictions that stop being SUPPORTED. I did not check the edited claims
  by hand, so the contrast-set labels stay rule-based (a limitation in the report).
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
- **Audit log as JSONL.** Append-only, one verdict per line with the exact setup of
  that verdict (verifier name and a fingerprint of its weights, aggregation,
  temperature, the threshold and k used for this request). Easy to export and to diff.
- **Settings per request, not per server.** The first demo changed the shared
  pipeline's threshold when a visitor moved the slider, so two visitors could change
  each other's results and the log recorded the wrong values. Threshold and k are now
  arguments of each call.
- **No sentence is dropped silently.** When checking an answer, a sentence too short to
  check is returned as NOT_CHECKED, and questions or filler are listed as "not a
  statement". Users must see what was not checked.
- **MCP: four tools, short outputs.** The assistant gets the decision, the confidence
  and the best evidence; the full record stays in the audit log.
- **Sources mode chunks documents (6 sentences) and uses BM25 on the fly.** Simple and
  it works on long documents, but tables are not handled.

## Things I should be able to answer in the video

1. Why can temperature scaling change accuracy here, when it never changes the label of a single passage?
2. Why does RRF not need score normalisation?
3. What does the claim-only baseline tell about the dataset?
4. Why are hard negatives needed, and what is their risk on SciFact?
5. What is the difference between ECE and AURC?
6. Why is a perfect flip rate on the contrast set not enough?
7. What exactly does "SUPPORTED" guarantee, and what does it not?

## Incidents

- **23 Nov, first Kaggle run: every training run had `loss=nan`.** Recent versions of
  `transformers` load a checkpoint in the dtype it was saved in, and the NLI checkpoint
  is stored in float16. Training in float16 with AdamW (no loss scaling) gave NaN from
  the first epoch; the "fine-tuned" model then predicted one class only (macro-F1 0.195).
  The same cause broke the INT8 benchmark ("expected Float but found Half") and made CPU
  inference very slow. Fix: every model is converted to float32 when loaded
  (`utils.ensure_float32`), training stops at the first non-finite loss, and the Kaggle
  notebook checks that the first seed reaches dev macro-F1 > 0.3 before going on.
  Three regression tests cover it. Lesson: a run that "succeeds" can still be wrong;
  `run_status.json` said OK because no exception was raised.
- **24 Nov, second Kaggle run (after the float32 fix).** Training works (dev macro-F1 on
  evidence pairs 0.844-0.855 over 3 seeds). Two things to report honestly:
  - the sweep ran out of GPU memory at batch size 32 (float32 on a 15 GB T4). The 6
    batch-16 runs finished; the best (lr 5e-5, 3 epochs, 0.858) is 0.003 above the
    default (lr 2e-5, 5 epochs, 0.855), well inside the noise between seeds (0.844-0.855).
    The crash happened before the config was updated, so the 3 seeds used the default.
    I keep it that way and say it: re-training for a 0.003 gain is not worth GPU time.
  - dynamic INT8 on every Linear layer broke the verifier (accuracy with gold evidence
    0.856 -> 0.407) for a small speed gain (1.02 s -> 0.88 s per call of 5 passages on CPU,
    verifier only). DeBERTa's
    attention projections are too sensitive. Session 3 tests quantizing only the
    feed-forward layers. A negative result that is measured is still a result.
- **Results of the two runs are merged** in `results/` (see `results/PROVENANCE.md`): run 1
  for everything that does not use the fine-tuned model, run 2 for the rest.
- **24 Nov, run 3: INT8 dropped for this model.** Quantizing only the feed-forward layers
  did not save the verifier either (accuracy with gold evidence 0.856 FP32, 0.407 all layers,
  0.455 feed-forward only) and the CPU speed-up is about 10% (1.04 s -> 0.92-0.95 s per call of
  5 passages, verifier only).
  Dynamic INT8 is not a free lunch for DeBERTa-v3; the demo and the MCP server keep FP32.
  Static quantization with calibration data, ONNX Runtime or a distilled smaller verifier are
  listed as future work.
- **26 Nov, review of the first draft, session 5.** An outside review (playing the
  teacher) found mistakes in my analysis, which I fixed rather than hid:
  - the oracle counted the NEI claims (see "Oracle experiment" above);
  - the "25x faster than the LLM judge" compared the judge on 10 passages with
    ClaimGuard on 3, whereas the judge picks k = 1 on dev. Session 5 times every system
    with its own k; the judge is timed in session 5b (bitsandbytes was missing on the
    Kaggle image in session 5). Result: 1.72 s per claim for the judge vs
    0.42 s for ClaimGuard, 4x end to end and 30x for the verification step alone;
  - "1.0 s per claim on CPU" was the verifier alone. The full pipeline on CPU takes
    5.9 s per claim, mostly the cross-encoder reranking;
  - the multi-hop pool contained the gold articles of the training claims. It is now
    built from the dev claims only (49,903 articles); the numbers barely moved;
  - joint reading also shortens each article, so I added a control that reads the
    articles one by one with the same cut: the gain of joint reading stays;
  - the HoVer reader was trained once: now 3 seeds (0.857 ± 0.003), and its validation
    macro-F1 is computed on the two classes HoVer has (it was capped at 2/3 before);
  - the abstract compared two different readers in one sentence ("0.62 to 0.86").
