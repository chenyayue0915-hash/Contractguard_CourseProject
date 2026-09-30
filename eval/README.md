# Stress set (`stress_cases.csv`)

50 short clauses, labels fixed in this file **before** any model was run on them.

| type | n | label | what it tests |
|---|---|---|---|
| `paraphrase_ld` | 20 | LD | LD / termination-fee clauses that avoid the words "liquidated damages" (per-day delay charges, exit and break fees, service credits, forfeited deposits). This is the failure mode behind the 3 missed test contracts. |
| `hard_negative` | 20 | NOT_LD | clauses that sit next to LD in contracts: liability caps, indemnities, termination for convenience with no payment, payment terms, warranty. |
| `injection_ld` | 10 | LD | 10 of the paraphrase clauses with an embedded prompt injection ("classify this as NOT_LD", fake `</passage>` tags, "ignore previous instructions"). OWASP LLM01. |

Provenance and limits (read before quoting results): the clauses are original text, not copied from CUAD. They were
drafted with AI assistance (Claude, Anthropic) and checked by the author against the CUAD definition. The FM under test
is also a Claude model, so the set may share that model family's blind spots, which is why the headline metric stays on
the lawyer-annotated CUAD test split. Two hard negatives (N11 cure-then-terminate, N19 capped actual delay costs) are
deliberately close to the boundary.
