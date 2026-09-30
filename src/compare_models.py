"""Business trade-off: run the SAME hybrid pipeline with each candidate model in config/models.json on a fixed
development subset, and put quality, cost and speed side by side. Test contracts are never used here.

    python src/compare_models.py                      # dev subset: 47 LD + 60 non-LD train contracts
    python src/compare_models.py --prompt few --full  # all 408 train contracts
    python src/compare_models.py --contracts-per-month 4000

Output: results/model_comparison.csv and results/model_comparison.md (paste-ready table for the report).
Needs OPENROUTER_API_KEY in .env for uncached calls. Rough cost: under US$1 per cheap model on the subset.
"""
import argparse, difflib, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import evaluate_hybrid as eh
from common import SEED
from config import load_models
from fm_verify import FMVerifier
from prices import price_table, live_prices
from pipeline import load_settings, candidates
from router import decide

RES = Path("results")

def subset(df, contracts, full, n_non=60):
    if full: return df, contracts
    ld = contracts[contracts.has_ld == 1].contract_id.tolist()
    non = sorted(contracts[contracts.has_ld == 0].contract_id); random.Random(SEED).shuffle(non)
    keep = set(ld) | set(non[:n_non])
    return df[df.contract_id.isin(keep)], contracts[contracts.contract_id.isin(keep)]

def prefetch(df, settings, fm, workers):
    """Make every FM call once, in parallel (results land in the cache); returns live-call latencies."""
    jobs = []
    for cid, g in df.groupby("contract_id", sort=False):
        texts, probs = g.text.tolist(), g.p.tolist()
        if max(probs) >= settings["T_BYPASS"]: continue            # bypassed contracts never call the FM
        jobs.append((cid, [(i, texts[i]) for i in candidates(texts, probs, settings["N_CAND"])]))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(lambda j: fm.verify(j[1], tag=j[0]), jobs))
    return [o["latency_s"] for o in outs if not o["cached"]], outs

def ml_only_row(df, contracts, settings):
    rows = []; has_ld = contracts.set_index("contract_id").has_ld
    for cid, g in df.groupby("contract_id", sort=False):
        probs, lab = g.p.tolist(), g.label.tolist()
        label, _ = decide(max(probs), None, settings["T_BYPASS"], settings["T_REVIEW_ML"])
        top = sorted(range(len(probs)), key=lambda i: -probs[i])[:5]
        rows.append({"contract_id": cid, "has_ld": int(has_ld[cid]), "label": label, "hit5": int(any(lab[i] for i in top)),
                     "fm_used": False, "fm_status": None, "cost_usd": 0.0, "latency_s": 0.0})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["zero", "few"], default="zero")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--contracts-per-month", type=int, default=4000, help="scenario: e.g. 1,000 SMEs x 4 contracts")
    ap.add_argument("--only", nargs="*", help="run only these model ids")
    a = ap.parse_args()
    cfg = load_models(); settings = load_settings()
    df, contracts = eh.scored_split("dev"); df, contracts = subset(df, contracts, a.full)
    n_ld = int(contracts.has_ld.sum()); print(f"dev subset: {len(contracts)} contracts, {n_ld} with LD")
    live, src = live_prices(refresh=True)
    if not live: print(f"(OpenRouter model list unavailable: {src}; using the price snapshot in config/models.json)")
    ptab = {r["id"]: r for r in price_table()[0]}
    rows = []
    base = ml_only_row(df, contracts, settings); s = eh.summarise(base, "ml_only")
    rows.append({"model": "ML only (no FM)", "label": "TF-IDF + LR", "prompt": "", **{k: s[k] for k in
                 ("hit@5", "hit@5_rate", "flag_precision", "flag_recall", "silent_miss", "abstention_rate")},
                 "cost_per_contract_usd": 0.0})
    for m in cfg["candidates"]:
        mid = m["id"]
        if not m.get("enabled", True) or (a.only and mid not in a.only): continue
        if live and mid not in live:
            print(f"SKIP {mid}: not on OpenRouter. Close matches: {difflib.get_close_matches(mid, list(live), n=3, cutoff=0.5)}")
            continue
        if live and not live[mid]["tools"]:
            print(f"SKIP {mid}: OpenRouter lists no tool-calling support for it"); continue
        fm = FMVerifier(model=mid, prompt=a.prompt, provider=cfg.get("provider", "openrouter"),
                        cache_path=RES / "fm_cache.jsonl", log_path=RES / "fm_calls.jsonl")
        if not fm.live and not fm.cache:
            from config import key_problem
            sys.exit(key_problem() or "OPENROUTER_API_KEY is missing.")
        print(f"\nRunning {mid} ({a.prompt}-shot) on {len(contracts)} contracts ...", flush=True)
        lat, outs = prefetch(df, settings, fm, a.workers)
        t, _ = eh.tune_t_review(df, contracts, settings, fm)
        res, _ = eh.run(df, contracts, dict(settings, T_REVIEW=t), fm)
        s = eh.summarise(res, mid); s.update(eh.fm_component(res, df))
        repairs = sum(1 for o in outs if o.get("repaired"))
        if not lat:                                               # everything cached: use logged latencies
            log = [json.loads(l) for l in open(RES / "fm_calls.jsonl")] if (RES / "fm_calls.jsonl").exists() else []
            lat = [r["latency_s"] for r in log if r.get("model") == mid and not r.get("cached")]
        calls = [o for o in outs]
        if lat:                                                   # real API latency, not the near-zero cache replays
            s["latency_p50_s"], s["latency_p95_s"] = float(np.median(lat)), float(np.percentile(lat, 95))
        pr = ptab.get(mid, {})
        rows.append({"model": mid, "label": m.get("label", mid), "tier": m.get("tier", ""), "prompt": a.prompt,
                     "official_in": pr.get("official_in"), "official_out": pr.get("official_out"),
                     "openrouter_in": pr.get("openrouter_in"), "openrouter_out": pr.get("openrouter_out"),
                     "price_note": pr.get("official_note", ""),
                     **{k: s.get(k) for k in ("hit@5", "hit@5_rate", "flag_precision", "flag_recall", "silent_miss",
                                              "abstention_rate", "fm_precision", "fm_recall", "fm_invalid_evidence_rate")},
                     "fm_call_failures": f"{s['fm_failures']}/{s['fm_calls']}", "fm_repairs": repairs, "T_REVIEW": t,
                     "cost_per_contract_usd": s["cost_per_contract_usd"],
                     "tokens_in_per_call": np.mean([o["in_tokens"] for o in calls]) if calls else 0,
                     "tokens_out_per_call": np.mean([o["out_tokens"] for o in calls]) if calls else 0,
                     "latency_p50_s": float(np.median(lat)) if lat else None,
                     "latency_p95_s": float(np.percentile(lat, 95)) if lat else None})
        print(eh.readable(s, f"{m.get('label', mid)} ({m.get('tier', '')})", a.contracts_per_month))
        print(f"  Answers that needed repair calls                 : {repairs}")
    out = pd.DataFrame(rows)
    # Keep rows from earlier runs (other models / prompts); a model re-run with the same prompt replaces its old row.
    prev_path = RES / "model_comparison.csv"
    if prev_path.exists() and not a.full:
        prev = pd.read_csv(prev_path)
        if "prompt" not in prev: prev["prompt"] = ""
        prev["prompt"] = prev["prompt"].fillna("")
        prev.loc[(prev.model != "ML only (no FM)") & (prev.prompt == ""), "prompt"] = "zero"   # rows from before this column
        keep = ~prev.set_index(["model", "prompt"]).index.isin(out.set_index(["model", "prompt"]).index)
        out = pd.concat([prev[keep], out], ignore_index=True)
    out = out.sort_values(["model"], key=lambda c: c.ne("ML only (no FM)")).reset_index(drop=True)
    out["monthly_fm_cost_usd"] = out.cost_per_contract_usd * a.contracts_per_month
    out["reviews_per_month"] = (out.abstention_rate * a.contracts_per_month).round()
    out.to_csv(prev_path, index=False)
    out["price_in_out"] = out.apply(lambda r: "" if pd.isna(r.get("openrouter_in")) else
                                    f"${r['openrouter_in']:.2f} / ${r['openrouter_out']:.2f}", axis=1)
    cols = ["label", "tier", "prompt", "price_in_out", "hit@5", "flag_precision", "silent_miss", "abstention_rate", "fm_precision", "fm_recall",
            "fm_call_failures", "cost_per_contract_usd", "monthly_fm_cost_usd", "reviews_per_month", "latency_p50_s", "latency_p95_s"]
    md = out[[c for c in cols if c in out]].copy()
    for c in md.columns:
        if c == "reviews_per_month": md[c] = md[c].map(lambda v: f"{int(v):,}")
        elif c == "monthly_fm_cost_usd": md[c] = md[c].map(lambda v: f"${v:,.2f}")
        elif c == "cost_per_contract_usd": md[c] = md[c].map(lambda v: f"${v:.4f}")
        elif md[c].dtype.kind == "f": md[c] = md[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    md = md.fillna("").replace("nan", "")
    header = (f"Dev subset: {len(contracts)} train contracts ({n_ld} with LD); each row lists its prompt; "
              f"scenario: {a.contracts_per_month:,} contracts per month. price_in_out = OpenRouter USD per 1M tokens "
              f"({src}); cost columns = what OpenRouter actually billed.\n\n")
    table = "| " + " | ".join(md.columns) + " |\n|" + "---|" * len(md.columns) + "\n" + \
            "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in md.values)
    (RES / "model_comparison.md").write_text(header + table + "\n")
    print("\nBaseline for comparison:")
    print(eh.readable(eh.summarise(base, "ml_only"), "ML only (TF-IDF + LR, no FM)", a.contracts_per_month))
    print("\nSaved: results/model_comparison.md (open it in VS Code and press Cmd+Shift+V to see it as a table)"
          " and results/model_comparison.csv")
