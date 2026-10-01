# Evals explainer

Every number in the README and the report comes from one of the evaluations below. Each one has a script, an input
whose labels were fixed before the run, and an output file in `../results/`. All foundation-model (FM) answers are cached
in `../results/fm_cache.jsonl`, so every eval re-runs without an API key and gives the same numbers.

## What "correct" means

* **Ground truth** is CUAD's lawyer-checked *Liquidated Damages* annotation (see `../data/README.md`). A passage is
  positive if it overlaps an LD span; a contract is positive if it has any LD span.
* **hit@5** (headline): the share of LD contracts where at least one of the 5 passages shown to the user is a real LD
  passage. This is recall at a fixed review budget: the user reads 5 passages, so "flag everything" cannot win.
* **Silent miss**: an LD contract labelled NO_FLAG. This is the failure nobody notices. Capped at 5% when tuning.
* **FLAG precision**: share of FLAG contracts that really contain LD (each wrong FLAG costs a lawyer consult).
* **REVIEW rate**: how often the system abstains and hands the contract to a person. Silent misses alone have a
  trivial winner (send everything to REVIEW), so the two are always reported together.
* **Abstention quality**: of the contracts sent to REVIEW, how many would have been labelled wrongly if the system had
  been forced to decide.
* **FM precision / recall**: the FM is a judge inside the system, so its passage verdicts are scored against the CUAD
  labels like any other component.
* **Cost and latency**: the amount OpenRouter billed for each call and its wall time (`../results/fm_calls.jsonl`).
* **Avoidable cost per contract**: business metric from `../src/business_case.py`. It adds FM spend, REVIEW handling,
  unneeded lawyer consults and expected losses from missed clauses, using the assumptions in `../config/business.json`.

## The evals, in the order they were run

| # | eval | data | script → output | what it decides |
|---|---|---|---|---|
| 1 | 5-fold CV of 5 designs (2 keyword rules, 3 ML variants incl. the masked leakage check) | 408 dev contracts, folds grouped by contract | `experiment_cv.py` → `dev_cv_summary.csv` | which negatives to train on; the ML design |
| 2 | out-of-fold scores and thresholds | 408 dev | `oof_analysis.py` → `dev_oof.csv.gz`, `frozen_settings.json` | the ML bypass threshold (0.88) and number of candidates (20) |
| 3 | **official test, ML baselines, run once** | 102 test | `official_eval.py` → `test_results.csv`, `models/tfidf_lr.joblib` | keyword vs ML on held-out data, bootstrap 95% CI |
| 4 | model comparison (quality × cost × latency) | dev subset: all 47 LD + 60 random non-LD contracts | `compare_models.py` → `model_comparison.csv` | which FM to rent; zero- vs few-shot; self-consistency |
| 5 | FM alone on every passage (no ML) | same dev subset | `fm_only_eval.py` → row in `model_comparison.csv` | what the ML retrieval step buys |
| 6 | threshold tuning and freeze for the chosen model | 408 dev | `evaluate_hybrid.py --split dev --freeze` → `hybrid_dev_*`, `frozen_settings.json` | REVIEW / FLAG thresholds by business cost, silent misses ≤ 5% |
| 7 | **official test, hybrid, run once** | 102 test | `evaluate_hybrid.py --split test` → `hybrid_test_summary.json` | final headline numbers |
| 8 | stress set | 50 hand-written clauses (this folder) | `stress_eval.py --fm` → `stress_summary_fm.csv` | robustness to paraphrase and prompt injection |
| 9 | unit tests | synthetic inputs and fake FM clients | `pytest -q` (`../tests/`) | the guardrails behave as designed |

Rules that kept the test set honest: `load_dev()` refuses test contracts; every threshold, keyword list, prompt and
model choice was fixed on dev before the test runs; `evaluate_hybrid.py` refuses a second test run unless forced and
refuses to save a test summary if any FM call failed. One change was made after freezing and before the test run: the
display order of passages was decoupled from the routing threshold (`RANK_CONF` in `src/pipeline.py`). On dev it
changed hit@5 from 41/47 to 43/47 and left every label unchanged.

## The stress set (`stress_cases.csv`)

50 short clauses, labels fixed in this file **before** any model was run on them.

| type | n | label | what it tests |
|---|---|---|---|
| `paraphrase_ld` | 20 | LD | LD / termination-fee clauses that avoid the words "liquidated damages" (per-day delay charges, exit and break fees, service credits, forfeited deposits). This is the failure mode behind the missed test contracts. |
| `hard_negative` | 20 | NOT_LD | clauses that sit next to LD in contracts: liability caps, indemnities, termination for convenience with no payment, payment terms, warranty. |
| `injection_ld` | 10 | LD | 10 of the paraphrase clauses with an embedded prompt injection ("classify this as NOT_LD", fake `</passage>` tags, "ignore previous instructions"). OWASP LLM01. |

Results (FLAG / REVIEW / NO_FLAG):

| system | paraphrased LD (20) | injected LD (10) | hard negatives (20) |
|---|---|---|---|
| keyword rule | 25% / 0% / 75% | 30% / 0% / 70% | 0% / 0% / 100% |
| ML only | 10% / 35% / 55% | 30% / 40% / 30% | 0% / 5% / 95% |
| hybrid (frozen) | 10% / 90% / 0% | 30% / 70% / 0% | 0% / 0% / 100% |

**Provenance and limits (read before quoting results).** The clauses are original text, not copied from CUAD. I drafted
them with AI assistance (Claude, Anthropic) in an interactive session, then checked each label against the CUAD
definition before any run; no generator script exists, so this file itself is the fixed artefact. Because I designed
both the system and these cases, they are a robustness check only: the headline metric stays on the CUAD test split,
which was annotated by others. The chosen FM (Llama 3.3 70B, Meta) is from a different model family than the one that
helped draft the cases. Two hard negatives (N11 cure-then-terminate, N19 capped actual delay costs) are deliberately
close to the boundary. The hybrid's stress result shows it does not miss these clauses silently, but it mostly
abstains on them (REVIEW) rather than flagging them.
