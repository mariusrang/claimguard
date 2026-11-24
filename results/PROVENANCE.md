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
