Monthly cost of each design for 4,000 contracts (USD). LD prevalence 12%; rates measured on the development subset, re-weighted to that prevalence.

| system | FM spend | REVIEW handling (owner time + escalations) | unneeded lawyer consults on FLAG | expected loss from missed LD | total avoidable cost | per contract |
|---|---|---|---|---|---|---|
| Llama 3.3 70B (open weights) (zero-shot) | $4.47 | $103,219 (1,906 reviews) | $0.00 (0) | $0.00 (0.0 missed) | **$103,224** | $25.81 |
| Claude Haiku 4.5 (zero-shot) | $31.25 | $120,768 (2,230 reviews) | $0.00 (0) | $0.00 (0.0 missed) | **$120,799** | $30.20 |
| Gemini 2.5 Flash-Lite (zero-shot) | $2.08 | $125,464 (2,316 reviews) | $0.00 (0) | $0.00 (0.0 missed) | **$125,466** | $31.37 |
| Gemini 2.5 Flash-Lite (few-shot) | $2.12 | $144,530 (2,668 reviews) | $0.00 (0) | $0.00 (0.0 missed) | **$144,533** | $36.13 |
| TF-IDF + LR | $0.00 | $93,274 (1,722 reviews) | $0.00 (0) | $204,255 (40.9 missed) | **$297,530** | $74.38 |

Sensitivity to the loss caused by one missed LD clause (total avoidable cost per month):

| system | loss $1,000 | loss $5,000 | loss $20,000 |
|---|---|---|---|
| Llama 3.3 70B (open weights) (zero-shot) | $103,224 | $103,224 | $103,224 |
| Claude Haiku 4.5 (zero-shot) | $120,799 | $120,799 | $120,799 |
| Gemini 2.5 Flash-Lite (zero-shot) | $125,466 | $125,466 | $125,466 |
| Gemini 2.5 Flash-Lite (few-shot) | $144,533 | $144,533 | $144,533 |
| TF-IDF + LR | $134,125 | $297,530 | $910,296 |

Sensitivity to the review escalation rate (share of REVIEW cases that still end in a paid lawyer consult):

| system | escalation 0% | escalation 25% | escalation 50% |
|---|---|---|---|
| Llama 3.3 70B (open weights) (zero-shot) | $7,944 | $103,224 | $198,503 |
| Claude Haiku 4.5 (zero-shot) | $9,321 | $120,799 | $232,277 |
| Gemini 2.5 Flash-Lite (zero-shot) | $9,653 | $125,466 | $241,279 |
| Gemini 2.5 Flash-Lite (few-shot) | $11,120 | $144,533 | $277,945 |
| TF-IDF + LR | $211,430 | $297,530 | $383,629 |

One REVIEW costs $54.17, so cutting the REVIEW rate by one percentage point saves about $2,167 a month; compare that with the FM spend column when judging a more expensive verifier.

Left out because every FM call failed (the model could not be used with forced tool calling): GPT-5.6 Luna.

Lawyer consults for contracts that really contain LD are needed in every design and are left out.

Assumptions (config/business.json):

- contracts_per_month: 4000 — ASSUMPTION: scenario of 1,000 SMEs x 4 vendor contracts per month
- ld_prevalence: 0.12 — CUAD v1: 61 of 510 contracts contain an LD clause (data in this repo)
- review_minutes_per_contract: 10 — ASSUMPTION: an owner reads the 5 highlighted passages and decides
- reviewer_cost_per_hour: 25 — ASSUMPTION: value of the owner's time per hour
- lawyer_consult_cost: 200 — ASSUMPTION: one short lawyer review of one flagged clause
- loss_per_missed_ld_clause: 5000 — ASSUMPTION, illustrative: problem statement cites a LegalShield study in which ~20% of small businesses lost over $5,000 to preventable legal problems
- review_escalation_rate: 0.25 — ASSUMPTION: share of REVIEW cases in which the owner, unsure after reading, still pays for a lawyer consult
