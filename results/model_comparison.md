Dev subset: 107 train contracts (47 with LD); each row lists its prompt; scenario: 4,000 contracts per month. price_in_out = OpenRouter USD per 1M tokens (live 2026-10-01 12:39 UTC); cost columns = what OpenRouter actually billed.

| label | tier | prompt | price_in_out | hit@5 | flag_precision | silent_miss | abstention_rate | fm_precision | fm_recall | fm_call_failures | cost_per_contract_usd | monthly_fm_cost_usd | reviews_per_month | latency_p50_s | latency_p95_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR |  |  |  | 41/47 | 1.00 | 4/47 | 0.47 |  |  |  | $0.0000 | $0.00 | 1,869 |  |  |
| GPT-5.6 Luna | mid | zero | $0.20 / $1.20 | 41/47 | 1.00 | 0/47 | 0.83 |  |  | 89/89 | $0.0000 | $0.00 | 3,327 | 0.06 | 0.11 |
| Gemini 2.5 Flash-Lite | budget | zero | $0.10 / $0.40 | 43/47 | 1.00 | 0/47 | 0.52 | 0.20 | 0.83 | 1/89 | $0.0005 | $2.08 | 2,093 | 1.83 | 2.41 |
| Gemini 2.5 Flash-Lite | budget | few | $0.10 / $0.40 | 42/47 | 1.00 | 0/47 | 0.58 | 0.17 | 0.84 | 0/89 | $0.0005 | $2.12 | 2,318 | 1.50 | 1.99 |
| Llama 3.3 70B (open weights) | open-weights | zero | $0.10 / $0.32 | 43/47 | 1.00 | 0/47 | 0.46 | 0.27 | 0.77 | 0/89 | $0.0011 | $4.47 | 1,832 | 19.97 | 35.64 |
| Claude Haiku 4.5 | mid | zero | $1.00 / $5.00 | 44/47 | 1.00 | 0/47 | 0.53 | 0.26 | 0.54 | 0/89 | $0.0078 | $31.25 | 2,131 | 5.49 | 11.29 |
| Gemini 2.5 Flash-Lite | budget | zero, every passage, no ML | $0.10 / $0.40 | 36/47 | 0.51 | 0/47 | 0.03 | 0.12 | 0.62 | 28/1180 | $0.0055 | $21.82 | 112 | 2.13 | 6.30 |
| Gemini 2.5 Flash-Lite | budget | zero x3 vote | $0.10 / $0.40 | 44/47 | 1.00 | 0/47 | 0.57 | 0.19 | 0.87 | 0/89 | $0.0013 | $5.34 | 2,280 | 3.11 | 10.47 |
