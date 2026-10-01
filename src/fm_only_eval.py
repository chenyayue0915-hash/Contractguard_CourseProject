"""The "prompt a rented model" rung: FM zero-shot classification of EVERY passage, with no ML retrieval and no tuned
thresholds (development subset only — the test set is not touched).

Why this script exists: the course asks to start with prompting a rented model and to climb to a more complex design
only when you can say what the extra rung buys. The hybrid sends ~20 ML-chosen candidates per contract in ONE call;
this baseline sends all ~160 passages of a contract in batches of 20 (about 8 calls per contract) and lets the FM
decide alone. Comparing the two rows answers "what does the narrow-ML retrieval step buy?" in quality, cost and calls.

    python src/fm_only_eval.py                                         # Gemini 2.5 Flash-Lite, zero-shot, dev subset
    python src/fm_only_eval.py --model meta-llama/llama-3.3-70b-instruct --workers 8

Decision rule (fixed in advance, nothing tuned): FLAG if any passage gets a valid LD answer (exact-quote evidence,
confidence >= 0.5); REVIEW if a call failed or the only LD answers are low-confidence or lack valid evidence; NO_FLAG
otherwise. The 5 passages shown are the FM's most confident LD passages.

Output: one row in results/model_comparison.csv (prompt = "zero, every passage, no ML"), so the row appears in
make_report.py's model table, in the business case and in figure 2. Calls are cached and logged like every FM call.
"""
import argparse, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import evaluate_hybrid as eh
from compare_models import subset
from config import load_models
from fm_verify import FMVerifier
from prices import price_table

RES = Path("results")
BATCH = 20
PROMPT_LABEL = "zero, every passage, no ML"

def batches(df):
    """(contract_id, batch_no, [(chunk_index, text), ...]) for every contract, in reading order."""
    jobs = []
    for cid, g in df.groupby("contract_id", sort=False):
        texts = g.text.tolist()
        for b, k in enumerate(range(0, len(texts), BATCH)):
            jobs.append((cid, b, [(i, texts[i]) for i in range(k, min(k + BATCH, len(texts)))]))
    return jobs

def decide_fm_only(answers):
    """answers: merged {chunk_index: record}; failed: True if any batch failed."""
    ld = {c: r for c, r in answers["results"].items() if r["label"] == "LD"}
    sure = {c: r for c, r in ld.items() if r["valid"] and r["confidence"] >= 0.5}
    if sure: return "FLAG", "fm_ld"
    if answers["failed"]: return "REVIEW", "fm_unavailable"
    if ld: return "REVIEW", "fm_uncertain"
    return "NO_FLAG", "fm_no_ld"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_models(); meta = next((m for m in cfg["candidates"] if m["id"] == a.model), {"label": a.model, "tier": ""})
    df, contracts = eh.scored_split("dev"); df, contracts = subset(df, contracts, full=False)
    fm = FMVerifier(model=a.model, prompt="zero", provider=cfg.get("provider", "openrouter"),
                    cache_path=RES / "fm_cache.jsonl", log_path=RES / "fm_calls.jsonl")
    if not fm.live and not fm.cache:
        from config import key_problem
        sys.exit(key_problem() or "OPENROUTER_API_KEY is missing.")
    jobs = batches(df)
    print(f"dev subset: {len(contracts)} contracts ({int(contracts.has_ld.sum())} with LD); {len(jobs)} FM calls of up to "
          f"{BATCH} passages with {a.model} ({a.workers} parallel workers; cached answers are free) ...", flush=True)
    done = [0]
    def one(j):
        o = fm.verify(j[2], tag=f"fmonly:{j[0]}:{j[1]}"); done[0] += 1
        if done[0] % 100 == 0 or done[0] == len(jobs): print(f"  FM calls done: {done[0]}/{len(jobs)}", flush=True)
        return o
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        outs = list(ex.map(one, jobs))

    per = {}
    for (cid, _, _), o in zip(jobs, outs):
        p = per.setdefault(cid, {"results": {}, "failed": False, "cost": 0.0, "calls": 0, "lat": []})
        p["results"].update(o["results"]); p["failed"] |= o["status"] != "ok"; p["cost"] += o["cost_usd"]
        p["calls"] += 1
        if not o["cached"]: p["lat"].append(o["latency_s"])
    has_ld = contracts.set_index("contract_id").has_ld; rows, chunk_rec = [], []
    for cid, g in df.groupby("contract_id", sort=False):
        p = per[cid]; lab, reason = decide_fm_only(p); touch = (g.ld_span_ids != "").astype(int).tolist()
        order = sorted(range(len(g)), key=lambda i: (
            not (i in p["results"] and p["results"][i]["label"] == "LD" and p["results"][i]["valid"]),
            -(p["results"][i]["confidence"] if i in p["results"] and p["results"][i]["label"] == "LD" else 0), i))
        rows.append({"contract_id": cid, "has_ld": int(has_ld[cid]), "label": lab, "reason": reason, "max_p": np.nan,
                     "hit5": int(any(g.label.iloc[i] for i in order[:5])), "fm_used": True,
                     "fm_status": "ok" if not p["failed"] else "partial", "cost_usd": p["cost"], "latency_s": sum(p["lat"])})
        for c, r in p["results"].items():
            chunk_rec.append({"pred": int(r["label"] == "LD" and r["valid"]), "raw_ld": int(r["label"] == "LD"),
                              "bad_ev": int(r["label"] == "LD" and not r["valid"]), "gold": touch[c]})
    res = pd.DataFrame(rows); s = eh.summarise(res, "fm_only")
    x = pd.DataFrame(chunk_rec); tp = int(((x.pred == 1) & (x.gold == 1)).sum())
    lat = [o["latency_s"] for o in outs if not o["cached"]]
    if not lat:                                                   # everything cached: use the logged API latencies
        import json
        log = [json.loads(l) for l in open(RES / "fm_calls.jsonl")] if (RES / "fm_calls.jsonl").exists() else []
        lat = [r["latency_s"] for r in log if str(r.get("tag", "")).startswith("fmonly:") and r.get("model") == a.model
               and not r.get("cached")]
    failed = sum(1 for o in outs if o["status"] != "ok")
    from business_case import load_business, avoidable_cost_per_contract
    c = {"ld_flag": s["flag_recall"], "ld_review": s["ld_review_rate"], "ld_silent": s["ld_silent_rate"],
         "non_flag": s["nonld_false_flag_rate"], "non_review": s["nonld_review_rate"]}
    avoid = avoidable_cost_per_contract(c, load_business()) + s["cost_per_contract_usd"]
    pr = {r["id"]: r for r in price_table()[0]}.get(a.model, {})
    row = {"model": a.model, "label": meta.get("label", a.model), "tier": meta.get("tier", ""), "prompt": PROMPT_LABEL,
           "openrouter_in": pr.get("openrouter_in"), "openrouter_out": pr.get("openrouter_out"),
           "official_in": pr.get("official_in"), "official_out": pr.get("official_out"),
           **{k: s[k] for k in ("hit@5", "hit@5_rate", "flag_precision", "flag_recall", "silent_miss", "abstention_rate",
                                "ld_review_rate", "ld_silent_rate", "nonld_review_rate", "nonld_false_flag_rate",
                                "cost_per_contract_usd")},
           "fm_precision": tp / max(int(x.pred.sum()), 1), "fm_recall": tp / max(int(x.gold.sum()), 1),
           "fm_invalid_evidence_rate": x.bad_ev.sum() / max(x.raw_ld.sum(), 1),
           "fm_call_failures": f"{failed}/{len(outs)}", "fm_repairs": sum(1 for o in outs if o.get("repaired")),
           "calls_per_contract": len(outs) / len(contracts), "avoidable_cost_per_contract": avoid,
           "tokens_in_per_call": float(np.mean([o["in_tokens"] for o in outs])),
           "tokens_out_per_call": float(np.mean([o["out_tokens"] for o in outs])),
           "latency_p50_s": float(np.median(lat)) if lat else None, "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
           "n_contracts": len(contracts), "n_ld": int(contracts.has_ld.sum())}
    path = RES / "model_comparison.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(prev):
        prev["prompt"] = prev.get("prompt", pd.Series([""] * len(prev))).fillna("")
        prev = prev[~((prev.model == a.model) & (prev.prompt == PROMPT_LABEL))]
    out = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    out["monthly_fm_cost_usd"] = out.cost_per_contract_usd * 4000
    out["reviews_per_month"] = (out.abstention_rate * 4000).round()
    out.to_csv(path, index=False)
    s.update(fm_precision=row["fm_precision"], fm_recall=row["fm_recall"], fm_failures=failed, fm_calls=len(outs),
             latency_p50_s=row["latency_p50_s"] or 0.0, latency_p95_s=row["latency_p95_s"] or 0.0)
    print(eh.readable(s, f"{meta.get('label', a.model)}: FM alone on every passage (no ML)"))
    print(f"  FM calls per contract                            : {len(outs) / len(contracts):.1f}")
    print(f"  Avoidable cost per contract (config/business.json): ${avoid:.2f}")
    print("Saved as a row of results/model_comparison.csv; run python src/business_case.py and python src/make_report.py.")
