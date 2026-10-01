# Data explainer

Everything ContractGuard learned from or was scored on is in this folder (or in `eval/` for the hand-written stress
set). Nothing here was collected from people, so no personal data is involved.

## Source and licence

| item | value |
|---|---|
| dataset | **CUAD v1**: Contract Understanding Atticus Dataset (Hendrycks et al., 2021) |
| link | https://github.com/TheAtticusProject/cuad (paper: https://arxiv.org/abs/2103.06268) |
| licence | CC BY 4.0, so redistribution with attribution is allowed. Copyright The Atticus Project. |
| what it is | 510 commercial contracts filed with the US SEC, annotated by law students and checked by lawyers for 41 clause types |
| clause type used | **Liquidated Damages**. CUAD's question for it: *"Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?"* |

**Does it measure what I need?** Partly. The CUAD label is exactly the clause type I flag (LD and termination fees), and
the annotations are lawyer-checked. But the contracts are US public-company filings, not SME vendor agreements. That
gap is stated as a limitation in the README and the report.

## Files

| file | rows | what it holds | made by |
|---|---|---|---|
| `raw/CUADv1.json` | 510 contracts | the original CUAD release (SQuAD format: contract text + annotated answer spans) | downloaded from the CUAD repository |
| `raw/test_titles.json` | 102 titles | the official CUAD **test** contracts (taken from CUAD's `test.json`); every other contract is development data | copied from CUAD |
| `processed/contracts.csv` | 510 | `contract_id`, `n_chars`, `has_ld` (1 if the contract has any LD span) | `src/prepare_data.py` |
| `processed/ld_spans.csv` | 121 | every LD span: `contract_id`, `span_id`, character `start`/`end`, `text` | `src/prepare_data.py` |
| `processed/chunks.csv.gz` | 79,922 | every 100-word passage (stride 50) of every contract: `text`, character offsets, `label` (1 = overlaps an LD span), `bucket`, `ld_span_ids` | `src/prepare_data.py` |

Rebuild the processed files at any time: `python src/prepare_data.py` (deterministic).

## Counts

| | contracts | with LD | LD spans | role |
|---|---|---|---|---|
| official train split | 408 | 47 | 98 | **development**: 5-fold CV, threshold tuning, model choice |
| official test split | 102 | 14 | 23 | **test**: scored once per system with frozen settings |
| total | 510 | 61 (12%) | 121 | |

Chunk buckets (all 510 contracts): 220 `pos` (overlap an LD span), 19,906 `hard_neg` (overlap another CUAD clause type),
59,796 `boiler_neg` (overlap no annotation at all).

## Design decisions made on this data

* **Negatives include unannotated boilerplate.** My Milestone-1 plan sampled negatives only from other annotated
  clauses. The instructor's feedback pointed out that most of a real contract is unannotated text. Training on every
  chunk cut false-positive passages per non-LD contract from 7.1 to 0.07 in development CV (table 1b in
  `results/report_tables.md`).
* **Split by contract, never by chunk.** Overlapping chunks of one contract never sit on both sides of a fold
  (`StratifiedGroupKFold` grouped by contract). `src/common.py: load_dev()` raises an error if a test contract appears.
* **Leakage check.** The literal words "liquidated damages" are a near-giveaway, so the same model was retrained with
  them masked. Before/after on test: hit@5 11/14 → 8/14 (dev CV 0.88 → 0.75). The model still finds LD clauses
  without the label words, but they carry real signal; this is why paraphrases are in the stress set.
* **Prevalence.** 12% of CUAD contracts contain LD. The business case re-weights every rate to this prevalence instead
  of reusing numbers from the LD-enriched development subset.

## Other data in the repository

* `../demo/`: two CUAD test contracts (excerpts, CC BY 4.0) and one **fictional** catering supply contract that I
  wrote, with and without an injected instruction. Used for the app demo only, not for scoring.
* `../eval/stress_cases.csv`: 50 hand-written clauses (see `../eval/README.md`).
* `../results/fm_cache.jsonl`: every foundation-model answer, so all numbers can be reproduced without an API key.
