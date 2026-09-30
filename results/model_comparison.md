Dev subset: 107 train contracts (47 with LD), prompt: zero-shot, scenario: 4,000 contracts per month. price_in_out = OpenRouter USD per 1M tokens (live 2026-09-30 07:18 UTC); cost columns = what OpenRouter actually billed.

| label | tier | price_in_out | hit@5 | flag_precision | silent_miss | abstention_rate | fm_precision | fm_recall | fm_call_failures | cost_per_contract_usd | monthly_fm_cost_usd | reviews_per_month | latency_p50_s | latency_p95_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR |  |  | 41/47 | 1.00 | 4/47 | 0.47 |  |  |  | $0.0000 | $0.00 | 1,869 |  |  |
| Gemini 2.5 Flash-Lite | budget | $0.10 / $0.40 | 43/47 | 0.60 | 0/47 | 0.06 | 0.20 | 0.83 | 1/89 | $0.0005 | $2.09 | 224 | 3.17 | 7.58 |
