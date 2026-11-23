# ClaimGuard — project plan (v2, after the expert review)

NLP final project, topic 2: **Fact Verification with Evidence Retrieval**.

**Deadline: 15 December 2026.** Compute: Google Colab only (no school GPU), so the
verifier stays DeBERTa-v3-base and the LLM judge is a quantized open model
(confirmed with the professor on 21 November).

## 1. One-sentence pitch

ClaimGuard checks if a claim is supported by documents, shows the evidence, gives a
confidence that can be trusted, and says "I don't know" when it is not sure. It runs
locally with a small model, and it plugs into Claude through an MCP server that keeps
an audit log.

Important limit, said from the start: ClaimGuard measures **faithfulness to the
documents**, not truth in the real world. If the document is wrong, a claim that
repeats it is still "SUPPORTED".

## 2. What changed after the review

I showed the first plan to an expert debate (a solution engineer at a model vendor and
a head of generative AI in a bank). Their main points, and what I did with them:

| Feedback | Decision |
| --- | --- |
| "Small specialised model vs big LLM judge" is already shown by MiniCheck (Tang et al., 2024). It cannot be my main claim. | Agreed. The LLM judge stays, but only as one baseline. My angle is **calibration + abstention, stress tests on numbers, and deployment with an audit trail**. |
| 7 bricks + 8 extensions is too much for one person. Better 3 things at 100% than 7 at 70%. | Agreed. Scope cut (see section 4). |
| The learned aggregator will overfit on SciFact (about 300 test claims). | Cut. I compare three rule-based aggregations instead. Listed as future work. |
| Fine-tuning the bi-encoder is not worth it against a recent embedding model. | Cut. Dense retrieval uses BGE-small as is, measured in ablation A1. |
| DeBERTa-large and a fine-tuned Flan-T5 for decomposition are not needed. | Cut. DeBERTa-v3-base only; decomposition is a sentence splitter with filters. |
| The errors that hurt in a bank are amounts, dates and thresholds. The contrast set should be a core part. | Promoted from "if time allows" to a main experiment. |
| "English only" is a blocker for a French bank. | Added a small French banking test set (60 claims, mostly numeric edits) and a comparison English verifier vs multilingual verifier. |
| A bank wants a trace: claim, evidence, score, decision, date, model version. | Added an audit log (JSONL), shown in the demo and exposed by the MCP server. |
| If the LLM already gives citations, just check the citations. | Added a `verify_citations` mode. |
| Say clearly that it measures faithfulness, not truth. | Written in the README, the demo, the MCP tool descriptions and the report. |
| Holm-Bonferroni and 200 annotated errors are overkill. | Kept bootstrap CI, McNemar and permutation tests. Error analysis is 100+ errors (the course minimum is 100, so I do not go under). |
| The hosted Space and the automatic hook are optional. | Cut. Docker is enough. The hook is future work. |

One thing I did **not** follow: the review suggested about 80 annotated errors. The
course guidelines ask for at least 100, so I keep 100+.

## 3. Final scope

1. **A calibrated verifier that can abstain.** Retrieval (BM25 / dense / hybrid) →
   cross-encoder reranking → DeBERTa-v3-base verifier → multi-evidence aggregation →
   temperature scaling → abstention threshold. Headline metric: *alert precision*
   (out of 10 alerts, how many are real problems).
2. **Stress tests on what hurts in practice.** A rule-based contrast set (numbers,
   negation, antonyms), a French banking test set, and Climate-FEVER as an
   out-of-domain test.
3. **Checking LLM answers (the extension the professor approved).** An answer is split
   into claims and each one goes through the same pipeline against its source
   document. Evaluated on LLM-AggreFact (real summaries and RAG answers written by
   LLMs, labelled by humans), with balanced accuracy, its official metric. The main
   pipeline stays the core of the project; this is the complement.
4. **A deployable tool.** Gradio demo, Docker image, MCP server with four tools
   (`verify_claim`, `verify_answer`, `verify_citations`, `get_audit_log`), audit log,
   INT8 quantization with measured size / latency / accuracy.

## 4. Pipeline

| # | Step | Method | Written by me |
| --- | --- | --- | --- |
| 1 | First-stage retrieval | BM25 (own implementation), dense BGE-small (exact cosine search), hybrid RRF | BM25, RRF, index + cache |
| 2 | Reranking | MS MARCO MiniLM cross-encoder on the top-50; also picks the 3 best sentences per abstract | integration, sentence selection |
| 3 | Verification | DeBERTa-v3-base, starts from a public NLI checkpoint, fine-tuned on SciFact | training pairs, training loop, label mapping |
| 4 | Aggregation | majority vote / most confident document / noisy-OR | all three |
| 5 | Calibration | temperature scaling on dev, ECE, risk-coverage, threshold picked on dev | all |
| 6 | Deployment | Gradio, Docker, MCP server, audit log, INT8 | all |

No commercial API anywhere. Pre-trained open models are starting points only.

## 5. Data

| Dataset | Use | Size |
| --- | --- | --- |
| SciFact (Wadden et al., 2020) | main benchmark: retrieval + verification, all ablations | about 5.2k abstracts, 1.1k labelled claims (official dev = my test) |
| Climate-FEVER (Diggelmann et al., 2020) | out-of-domain test of the verifier | 1,535 claims (1,381 after dropping DISPUTED) |
| Contrast set (built by rules from SciFact test) | robustness to number / negation / antonym edits | depends on the test split, a few hundred |
| French bank set (built for this project) | cross-lingual + numeric stress test | 60 claims on 20 invented documents |
| LLM-AggreFact (Tang et al., 2024) | extension: sentences written by LLMs | 100 claims per sub-dataset (subsample, Colab budget) |

## 6. Experiments

**Baselines (5):** majority class · claim-only TF-IDF + logistic regression (no
evidence, shows dataset bias) · rule-based lexical verifier · BM25 + zero-shot NLI
(no fine-tuning) · open-source LLM as zero-shot judge.

**Ablations (6):**

| # | Question | Variants |
| --- | --- | --- |
| A1 | Which retriever? | BM25 / dense / hybrid RRF |
| A2 | Does the reranker help? | with / without (retrieval metrics and end-to-end) |
| A3 | How many evidence passages? | k = 1, 3, 5, 10 |
| A4 | How to aggregate? | majority / max-confidence / noisy-OR |
| A5 | Where does the verifier quality come from? | zero-shot NLI / SciFact fine-tuning from a plain encoder / NLI pre-training + SciFact fine-tuning |
| A6 | Retrieval or verification errors? | retrieved evidence vs gold ("oracle") evidence |

**Hyperparameter search:** learning rate {1e-5, 2e-5, 5e-5} × batch size {16, 32} ×
epochs {3, 5}, on dev (12 runs of about 3 minutes on a Colab T4).

**Choices made on dev only:** number of evidence passages k, aggregation strategy and
temperature are chosen together on dev for every system (baselines included); the test
set is only used to report.

**Statistics:** 3 seeds (mean ± std), bootstrap 95% CI on every headline number,
McNemar test between systems, paired permutation test for retrieval.

**Error analysis (100+ errors):** all errors of the best system on the SciFact test
set, completed with errors on Climate-FEVER to pass 100. Each error is first tagged
automatically (retrieval miss or verification error, has a number, has a negation),
then given one category by hand. I annotate 25 of them a second time one week later
and report Cohen's kappa.

Starting taxonomy: evidence not retrieved · wrong sentence selected · numeric
reasoning · negation / quantifier · needs several pieces of evidence · REFUTED vs NEI
confusion · domain vocabulary · questionable gold label.

## 7. Deliverables and where they are

| Guideline | Where |
| --- | --- |
| Code repository with the required structure | this repo (`data/`, `models/`, `src/`, `notebooks/`, `configs/`, `tests/`, `app.py`, `Dockerfile`, `requirements.txt`, `README.md`, `report.pdf`) |
| README with all required sections | `README.md` |
| Training command with a config file | `python scripts/train_verifier.py --config configs/train_verifier.yaml` |
| Interactive demo (text input, upload, real time, visualisation, examples, error handling, inference time, model size) | `app.py` |
| Docker | `Dockerfile`, `make docker`, `make docker-run` |
| Unit tests (preprocessing, forward pass, metrics) | `tests/`, `make test` |
| Docstrings, comments, type hints | everywhere in `src/` |
| Reproducibility (seeds, requirements, configs) | `set_seed`, `requirements.txt`, `configs/` |
| Technical report, 5–10 pages | `report/report_draft.md` → `report.pdf` |
| 10–15 minute video | to record at the end |

## 8. Status

| Step | Status |
| --- | --- |
| Code: pipeline, stress tests, LLM-AggreFact evaluation, demo, MCP server | done, 63 tests pass offline |
| French bank set, toy sample, contrast-set builder | done (French set still to re-read by hand) |
| Experiments on SciFact, Climate-FEVER, LLM-AggreFact | to run on Colab |
| Error annotation, report results, video | after the runs |

## 9. Schedule to 15 December

| Week | Dates | Goal | Done when |
| --- | --- | --- | --- |
| 1 | 21–27 Nov | Colab sessions 1 and 2 (data, retrieval, baselines, sweep, 3 seeds). Re-read the 60 French items. Read `src/` and `DECISIONS.md`. | `results/` has retrieval, baselines, sweep and 3 checkpoints on Drive |
| 2 | 28 Nov–4 Dec | Colab session 3 (pipeline + ablations, calibration, LLM judge, stress tests, LLM-AggreFact). Export errors, **first annotation pass (100+)**. Hand-check 50 contrast examples. | `make tables` gives every table; `errors_annotated.csv` has 100+ rows |
| 3 | 5–11 Dec | Write the result sections and the error analysis from `results/tables.md`. **Second annotation pass on 25 errors** (one week after the first) for Cohen's kappa. Build the Docker image once on my laptop. Check the 18 references. | `report.pdf` complete, 5–10 pages |
| 4 | 12–14 Dec | Record the 10–15 min video, last README pass, final zip / repo. | everything sent |
| — | 15 Dec | Deadline. Video to benjamin.dallard@centralesupelec.fr. | |

Rule: if week 2 slips, drop the contrast-set hand check and Climate-FEVER first.
The main pipeline, the LLM-AggreFact extension, calibration and the error analysis
are the parts the professor asked for and the grid rewards most.
