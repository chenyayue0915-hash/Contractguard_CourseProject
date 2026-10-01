# Results folder: what each file is

Every file here is written by a script in `src/`; none is edited by hand. The report numbers come from
`report_tables.md`, which `make_report.py` rebuilds from the other files. See `../eval/README.md` for what each
evaluation means.

## Report

| file | made by | content |
|---|---|---|
| `report_tables.md` | `make_report.py` | every table used in the report (ladder, data design, abstention, model choice, prompt techniques, cost, stress, build vs buy) |
| `../figures/fig1–3_*.png` | `make_report.py` | technique ladder on test, model trade-off, monthly cost |

## Development data only (408 train contracts)

| file | made by | content |
|---|---|---|
| `dev_cv_folds.csv`, `dev_cv_summary.csv` | `experiment_cv.py` | 5-fold CV of the 5 designs, per fold and mean ± std |
| `dev_oof.csv.gz` | `oof_analysis.py` | out-of-fold ML score of every dev chunk (input to all dev FM runs) |
| `dev_candidates.csv` | `oof_analysis.py` | how many LD contracts the top-N ML candidates cover (chose N = 20) |
| `dev_threshold_sweep.csv` | `oof_analysis.py` | ML-only threshold sweep (chose the 0.88 bypass and the ML-only REVIEW threshold) |
| `frozen_settings.json` | `oof_analysis.py`, then `evaluate_hybrid.py --freeze` | every threshold, the prompt and the model, fixed before the test runs |
| `model_comparison.csv` / `.md` | `compare_models.py`, `fm_only_eval.py` | one row per model × prompt technique on the 107-contract dev subset |
| `hybrid_dev_zero_meta-llama--llama-3.3-70b-instruct_s1_*` | `evaluate_hybrid.py --split dev --freeze` | chosen system on all 408 dev contracts: per-contract labels, summary, threshold sweep |
| `business_case.csv` / `.md` | `business_case.py` | monthly cost to serve and sensitivity tables |

## Official test set (102 contracts, each system scored once)

| file | made by | content |
|---|---|---|
| `test_results.csv` | `official_eval.py` | keyword and ML baselines, with bootstrap 95% CI |
| `test_scores.csv.gz` | `official_eval.py` | ML score of every test chunk (frozen model) |
| `hybrid_test_summary.json` | `evaluate_hybrid.py --split test` | headline hybrid result |
| `hybrid_test_zero_meta-llama--llama-3.3-70b-instruct_s1_contracts.csv` | same | per-contract label and reason code on test |

## Stress set (50 hand-written clauses)

| file | made by | content |
|---|---|---|
| `stress_summary.csv`, `stress_cases_scored.csv` | `stress_eval.py` (no key) | keyword and ML-only results |
| `stress_summary_fm.csv`, `stress_cases_scored_fm.csv` | `stress_eval.py --fm` | the same plus the frozen hybrid (used in the report) |

## FM calls and prices

| file | made by | content |
|---|---|---|
| `fm_cache.jsonl` | `fm_verify.py` | every FM answer, so all runs reproduce without a key or spend |
| `fm_calls.jsonl` | `fm_verify.py` | log of every call: model, prompt, tokens, billed cost, latency, status. It also keeps the failed early attempts (empty answers before the repair logic, GPT-5.6 Luna's HTTP 404s) as an audit trail |
| `price_table.md` | `prices.py --refresh` | candidate models sorted by estimated cost per contract |

Not in the repository (git-ignored, local only): `fm_debug.jsonl` (raw output of empty FM answers, used to debug
them) and `openrouter_prices_live.json` (24-hour price cache).
