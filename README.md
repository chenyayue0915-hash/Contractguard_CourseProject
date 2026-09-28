# ContractGuard — flags liquidated-damages clauses before SMEs sign

Flag only: no rewriting, no legal advice. Data: [CUAD v1](https://github.com/TheAtticusProject/cuad) (CC BY 4.0), official train/test split.

## Reproduce (about 5 min on a laptop CPU)
```bash
pip install -r requirements.txt
# put CUADv1.json in data/raw/  (data/raw/test_titles.json = the 102 official test contracts, from CUAD test.json)
python src/prepare_data.py      # chunk every contract, label pos / hard_neg / boiler_neg
python src/experiment_cv.py     # DEV: 5-fold CV on the 408 train contracts, compares 5 designs
python src/oof_analysis.py      # DEV: out-of-fold scores -> thresholds, candidate N -> results/frozen_settings.json
python src/official_eval.py     # FINAL: test (102) scored once with frozen settings; saves models/tfidf_lr.joblib
```

## Train / test discipline
`src/common.py` owns the split. `load_dev()` returns only the 408 train contracts and asserts that no test contract is present;
`experiment_cv.py` and `oof_analysis.py` use nothing else. Only `official_eval.py` calls `load_test()`, after reading settings
frozen on dev from `results/frozen_settings.json`. Keyword lists are fixed in `common.py` and were not edited after any test run.

## Files
| file | role |
|---|---|
| `src/chunker.py` | 100-word windows, stride 50; the same function is used in training, evaluation and the app |
| `src/prepare_data.py` | labels every chunk: `pos` / `hard_neg` (another CUAD category) / `boiler_neg` (unannotated) |
| `src/common.py` | loaders, official split + leakage guard, keyword rules, model factory, metrics |
| `src/experiment_cv.py` | dev CV: keyword vs Milestone-1 design vs chunk-level vs masked |
| `src/oof_analysis.py` | dev out-of-fold scores: hit@k, threshold sweep, bypass threshold, candidate coverage |
| `src/official_eval.py` | one-shot test evaluation with bootstrap CI; trains the deployable model |

## Results
Dev (5-fold CV, 408 train contracts, 47 with LD), mean ± sd across folds:

| model | hit@5 | span R@5 | FP chunks per non-LD contract (thr 0.5) |
|---|---|---|---|
| keyword "liquidated damages" | 0.51 ± 0.28 | 0.48 | 0.01 |
| keyword broad | 0.68 ± 0.12 | 0.60 | 1.24 |
| M1: span negatives only (Milestone-1 design) | 0.83 ± 0.13 | 0.73 | 7.08 |
| **M2: chunk-level training** | **0.88 ± 0.11** | **0.79** | **0.07** |
| M2 with label words masked | 0.75 ± 0.14 | 0.65 | 0.16 |

Official test (102 contracts, 14 with LD, 23 spans), scored once:

| model | hit@5 (95% CI) | span R@5 | contract recall | contract precision | FP chunks / non-LD contract |
|---|---|---|---|---|---|
| keyword "liquidated damages" | 8/14 [0.29, 0.79] | 0.57 | 0.50 | 1.00 | 0.00 |
| keyword broad | 8/14 [0.36, 0.86] | 0.52 | 0.79 | 0.28 | 1.08 |
| M1 span negatives | 8/14 [0.29, 0.86] | 0.65 | 0.93 | 0.18 | 3.76 |
| **M2 chunk level** | **11/14 [0.57, 1.00]** | **0.78** | 0.57 | 0.89 | 0.02 |
| M2 masked | 8/14 [0.29, 0.79] | 0.57 | 0.50 | 0.58 | 0.07 |

Frozen on dev: `T_BYPASS = 0.88` (ML alone may FLAG), `N_CAND = 20`. Candidates (top-20 plus keyword hits) cover 45/47 LD
contracts on dev and 14/14 on test, about 20 chunks per contract.
