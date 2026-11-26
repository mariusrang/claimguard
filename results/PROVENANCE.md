# Where the result files come from

Two Kaggle runs (GPU T4), 23-24 Nov 2026.

- Run 1 (full notebook 03): every training run had loss=nan (checkpoint loaded in float16, see DECISIONS.md, Incidents).
  Kept from run 1 only what does not use the fine-tuned verifier: retrieval.json, pipeline_bm25_zeroshot,
  pipeline_zeroshot, pipeline_llm_judge, aggrefact_zeroshot, aggrefact_lexical, given_french_multilingual.
- Run 2 (notebook 04, after the float32 fix): everything that uses the fine-tuned verifier (sweep, 3 seeds,
  calibration, main pipeline + ablations, plain encoder, stress tests, efficiency, aggrefact).
  The sweep ran the 6 batch-size-16 settings; batch size 32 ran out of GPU memory in float32 on the T4.
- Rule-based results (pipeline_lexical, given_*_lexical, calibration_lexical, baselines) are identical in
  every run (no randomness, no GPU).
- Run 3 (notebook 05, 24 Nov, same trained verifier as run 2, loaded from the run-2 output): given_climate_zeroshot,
  given_contrast_zeroshot, efficiency.json (3 INT8 recipes), and per-example predictions on Climate-FEVER and the
  contrast set. given_climate / given_contrast were re-computed and are identical to run 2 (deterministic inference).
- Session 4 (26 Nov morning, notebook 06): first multi-hop run. Its pool also contained the gold
  articles of the training claims (56,417 articles), which the review pointed out. It is REPLACED by
  session 5 below; the old numbers are only kept in the git history.
- Session 5 (26 Nov afternoon, notebook `notebooks/07_kaggle_session5.ipynb`, T4, ~75 min), same
  trained SciFact verifier as run 2 (session-2 output as input; `predictions.jsonl` and `pipeline.json`
  in the session-5 zip are byte-identical to the ones here, md5 checked):
  - `oracle.json`, `oracle_predictions.jsonl`: the verifier re-run on gold abstracts (our sentence
    selection) and on gold sentences, compared on the 188 test claims that have evidence.
  - `latency_cpu.json`: full pipeline, CPU only (CUDA hidden), 2 threads, 20 claims after 3 warm-up.
    The GPU latency step failed for the 4-bit LLM judge (bitsandbytes missing on the image); the three
    other GPU timings are in the notebook log only, and are re-measured in session 5b.
  - `multihop.json`, `figures/multihop.png`, `hover_joint_reader*_train_log.json`: multi-hop re-run with
    the pool built from the DEV claims only (their TF-IDF top-100 + their gold articles, 49,903 articles),
    the truncation control, the HoVer reader trained with seeds 13 / 42 / 2026, and the 2 x 2
    end-to-end table at top 5 and top 10.
  - `hover_claim_only.json`: claim-only baseline; the same file was also produced locally (CPU,
    same code and data) with the same numbers.
- `paired_tests.json`: computed locally (CPU) from the saved predictions of run 2 and run 1
  (`scripts/paired_tests.py`), each system with its own k, aggregation and temperature chosen on dev.
- Session 5b (26 Nov, notebook `notebooks/08_kaggle_llm_latency.ipynb`, T4): `latency_cuda.json`.
  The main system and the Qwen-7B judge were timed in the same session (bitsandbytes installed),
  each at its own k chosen on dev. The zero-shot and BM25 + zero-shot timings in the same file
  come from the session-5 log (the file itself was not written there, see above).
