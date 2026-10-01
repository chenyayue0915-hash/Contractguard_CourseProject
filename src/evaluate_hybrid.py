"""Hybrid evaluation (ML retrieval + FM verification + decision rules).

DEV (408 train contracts, out-of-fold ML scores):
    python src/evaluate_hybrid.py --split dev --prompt zero          # prompt ladder rung 1
    python src/evaluate_hybrid.py --split dev --prompt few           # rung 2
    python src/evaluate_hybrid.py --split dev --prompt few --samples 3   # optional: self-consistency (x3 cost)
    python src/evaluate_hybrid.py --split dev --prompt zero --model meta-llama/llama-3.3-70b-instruct --freeze
                                                                     # write thresholds / prompt / model to frozen_settings.json
TEST (102 contracts) — once, with frozen settings:
    python src/evaluate_hybrid.py --split test

The FM is called through OpenRouter by default (key OPENROUTER_API_KEY in the git-ignored .env file);
--model takes any OpenRouter model id, e.g. anthropic/claude-haiku-4.5 or openai/gpt-5-mini.
To pick a model on cost vs quality first, run src/compare_models.py (dev subset, all candidates).
Costs about US$3 per full dev run with a Haiku-class model; cached calls are free and reproduce exactly.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from common import SEED, load_dev, load_test
from fm_verify import FMVerifier, DEFAULT_MODEL, DEFAULT_PROVIDER, KEY_ENV
from pipeline import analyze_scored, load_settings, candidates

RES = Path("results")

def scored_split(split):
    if split == "dev":
        chunks, contracts, spans = load_dev()
        p = pd.read_csv(RES / "dev_oof.csv.gz")[["contract_id", "chunk_id", "oof"]].rename(columns={"oof": "p"})
    else:
        chunks, contracts, spans = load_test()
        p = pd.read_csv(RES / "test_scores.csv.gz")[["contract_id", "chunk_id", "p"]]
    df = chunks.merge(p, on=["contract_id", "chunk_id"], how="left")
    assert df.p.notna().all(), "missing ML scores: run oof_analysis.py (dev) / official_eval.py (test) first"
    df["touch"] = (df.ld_span_ids != "").astype(int)
    return df.sort_values(["contract_id", "chunk_id"]), contracts

def run(df, contracts, settings, fm):
    rows, cand_rows = [], []; has_ld = contracts.set_index("contract_id").has_ld
    for cid, g in df.groupby("contract_id", sort=False):
        texts, probs = g.text.tolist(), g.p.tolist()
        out = analyze_scored(texts, probs, settings, fm, tag=cid)
        lab, touch = g.label.tolist(), g.touch.tolist()
        rows.append({"contract_id": cid, "has_ld": int(has_ld[cid]),
                     "label": out["label"], "reason": out["reason"], "max_p": out["max_p"],
                     "hit5": int(any(lab[t["chunk"]] for t in out["top"])),
                     "fm_used": out["fm_used"], "fm_status": out["fm_status"],
                     "cost_usd": out["cost_usd"], "latency_s": out["latency_s"]})
        rows[-1]["out_obj"] = out
        for i in out["candidates"]:
            cand_rows.append({"contract_id": cid, "chunk": i, "touch": touch[i]})
    return pd.DataFrame(rows), pd.DataFrame(cand_rows)

def prefetch(df, settings, fm, workers=4):
    """Make every FM call once, in parallel (answers land in the cache, so run() and tune() replay them);
    returns the latencies of the live calls."""
    jobs = []
    for cid, g in df.groupby("contract_id", sort=False):
        texts, probs = g.text.tolist(), g.p.tolist()
        if max(probs) >= settings["T_BYPASS"]: continue            # bypassed contracts never call the FM
        jobs.append((cid, [(i, texts[i]) for i in candidates(texts, probs, settings["N_CAND"])]))
    done = [0]
    def one(j):
        o = fm.verify(j[1], tag=j[0]); done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(jobs): print(f"  FM calls done: {done[0]}/{len(jobs)}", flush=True)
        return o
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        outs = list(ex.map(one, jobs))
    return [o["latency_s"] for o in outs if not o["cached"]], outs

def summarise(res, name):
    ld, non = res[res.has_ld == 1], res[res.has_ld == 0]
    flag = res.label == "FLAG"
    s = {"run": name, "contracts": len(res), "ld_contracts": len(ld),
         "hit@5": f"{int(ld.hit5.sum())}/{len(ld)}", "hit@5_rate": ld.hit5.mean(),
         "flag_precision": (flag & (res.has_ld == 1)).sum() / max(flag.sum(), 1),
         "flag_recall": (ld.label == "FLAG").mean(),
         "abstention_rate": (res.label == "REVIEW").mean(),
         "silent_miss": f"{int((ld.label == 'NO_FLAG').sum())}/{len(ld)}",
         "ld_not_flagged_that_went_to_review": ((ld.label == "REVIEW").sum() / max((ld.label != "FLAG").sum(), 1)),
         "ld_review_rate": (ld.label == "REVIEW").mean(), "ld_silent_rate": (ld.label == "NO_FLAG").mean(),
         "nonld_review_rate": (non.label == "REVIEW").mean(),
         "nonld_false_flag_rate": (non.label == "FLAG").mean(),
         "fm_calls": int(res.fm_used.sum()), "fm_failures": int((res.fm_used & (res.fm_status != "ok")).sum()),
         "cost_per_contract_usd": res.cost_usd.mean(),
         "latency_p50_s": float(res[res.fm_used].latency_s.median()) if res.fm_used.any() else 0.0,
         "latency_p95_s": float(res[res.fm_used].latency_s.quantile(0.95)) if res.fm_used.any() else 0.0}
    s["label_table"] = pd.crosstab(res.has_ld, res.label).to_dict()
    return s

def readable(s, title, per_month=4000):
    """Plain-text summary for the terminal (the .md / .json files keep the full numbers)."""
    pct = lambda v: "" if v is None or v != v else f"{100 * v:.0f}%"
    ld = s.get("ld_contracts") or int(str(s.get("hit@5", "0/0")).split("/")[1] or 0)
    lines = [f"== {title}",
             f"  LD contracts found in the 5 passages shown (hit@5) : {s.get('hit@5')}",
             f"  LD contracts missed silently (NO_FLAG)           : {s.get('silent_miss')}",
             f"  FLAG precision (flags that were real LD)         : {pct(s.get('flag_precision'))}",
             f"  Sent to a person (REVIEW rate)                   : {pct(s.get('abstention_rate'))}"
             f"  -> about {round((s.get('abstention_rate') or 0) * per_month):,} of {per_month:,} contracts a month"]
    if s.get("fm_calls"):
        lines += [f"  FM calls still failing after repair             : {s.get('fm_failures')}/{s.get('fm_calls')}",
                  f"  FM precision / recall on candidate passages      : {pct(s.get('fm_precision'))} / {pct(s.get('fm_recall'))}",
                  f"  Billed cost per contract                         : ${s.get('cost_per_contract_usd', 0):.4f}"
                  f"  -> ${s.get('cost_per_contract_usd', 0) * per_month:,.2f} per {per_month:,} contracts",
                  f"  Latency per FM call, median / 95th percentile    : {s.get('latency_p50_s', 0):.1f} s / {s.get('latency_p95_s', 0):.1f} s"]
    return "\n".join(lines)

def fm_component(res, df):
    """FM precision / recall on the candidate chunks it judged, against CUAD labels (chunk touches an LD span)."""
    rec = []
    for r in res.itertuples():
        fm = r.out_obj.get("fm_raw")
        if not fm: continue
        g = df[df.contract_id == r.contract_id].reset_index(drop=True)
        for c, v in fm["results"].items():
            rec.append({"pred": int(v["label"] == "LD" and v["valid"]), "raw_ld": int(v["label"] == "LD"),
                        "invalid_evidence": int(v["label"] == "LD" and not v["valid"]), "gold": int(g.touch[c])})
    if not rec: return {}
    x = pd.DataFrame(rec); tp = ((x.pred == 1) & (x.gold == 1)).sum()
    return {"fm_chunks_judged": len(x), "fm_precision": tp / max(x.pred.sum(), 1), "fm_recall": tp / max(x.gold.sum(), 1),
            "fm_invalid_evidence_rate": x.invalid_evidence.sum() / max(x.raw_ld.sum(), 1)}

def tune(df, contracts, settings, fm, max_silent=0.05,
         grid_review=np.round(np.arange(0.05, 0.90, 0.05), 2), grid_conf=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
         grid_flag_min=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    """Choose T_REVIEW, FM_CONF and T_FLAG_MIN on DEV by business cost (config/business.json):
    minimise owner review time + unneeded lawyer consults + expected loss from missed clauses, subject to at most
    `max_silent` of LD contracts ending as NO_FLAG. The FM is called once per contract; every threshold setting is then
    replayed on the stored answers, so tuning costs nothing extra."""
    from business_case import load_business, avoidable_cost_per_contract
    from router import decide
    cfg = load_business()
    res, _ = run(df, contracts, settings, fm)
    feats = [(r.has_ld, r.out_obj["max_p"], r.out_obj["fm_raw"]) for r in res.itertuples()]
    fm_cost = res.cost_usd.mean(); table = []
    for tr in grid_review:
        for fc in grid_conf:
            for tf in grid_flag_min:
                lab = [(h, decide(p, f, settings["T_BYPASS"], float(tr), fc, tf)[0]) for h, p, f in feats]
                ld = [l for h, l in lab if h == 1]; non = [l for h, l in lab if h == 0]
                c = {"ld_flag": ld.count("FLAG") / len(ld), "ld_review": ld.count("REVIEW") / len(ld),
                     "ld_silent": ld.count("NO_FLAG") / len(ld), "non_flag": non.count("FLAG") / len(non),
                     "non_review": non.count("REVIEW") / len(non)}
                table.append({"T_REVIEW": float(tr), "FM_CONF": fc, "T_FLAG_MIN": tf, **c,
                              "avoidable_cost_per_contract": avoidable_cost_per_contract(c, cfg) + fm_cost})
    t = pd.DataFrame(table)
    ok = t[t.ld_silent <= max_silent]
    best = (ok if len(ok) else t).sort_values(["avoidable_cost_per_contract", "T_REVIEW"], ascending=[True, False]).iloc[0]
    return {k: float(best[k]) for k in ("T_REVIEW", "FM_CONF", "T_FLAG_MIN")}, \
        t.sort_values("avoidable_cost_per_contract").reset_index(drop=True), float(best["avoidable_cost_per_contract"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--prompt", choices=["zero", "few"], default=None)
    ap.add_argument("--model", default=None, help="OpenRouter model id (default: config/models.json default_model)")
    ap.add_argument("--provider", choices=["openrouter", "anthropic"], default=None)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--freeze", action="store_true", help="dev only: store tuned T_REVIEW + prompt + model")
    ap.add_argument("--force-rerun-test", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel FM calls (slow models such as Llama need this)")
    a = ap.parse_args()
    settings = load_settings()
    if a.split == "test":
        if "T_REVIEW_frozen" not in settings:
            sys.exit("Freeze settings on dev first: python src/evaluate_hybrid.py --split dev --prompt few --freeze")
        if (RES / "hybrid_test_summary.json").exists() and not a.force_rerun_test:
            sys.exit("The hybrid was already scored on test. The test set is used once; rerun only with --force-rerun-test "
                     "and report that you did.")
        a.prompt, a.model, a.samples = settings["PROMPT"], settings["MODEL"], settings["SAMPLES"]
        a.provider = settings.get("PROVIDER", DEFAULT_PROVIDER)
        settings["T_REVIEW"] = settings["T_REVIEW_frozen"]
    from config import load_models
    a.prompt = a.prompt or "zero"; a.provider = a.provider or DEFAULT_PROVIDER
    a.model = a.model or load_models().get("default_model", DEFAULT_MODEL)
    fm = FMVerifier(model=a.model, prompt=a.prompt, samples=a.samples, provider=a.provider,
                    cache_path=RES / "fm_cache.jsonl", log_path=RES / "fm_calls.jsonl")
    if a.split == "test" and not fm.live:
        sys.exit(f"The one-shot test run needs {KEY_ENV[a.provider]} in .env (a run without it would only measure missing calls).")
    if not fm.live and not fm.cache:
        from config import key_problem
        sys.exit(key_problem(KEY_ENV[a.provider]) or "API key missing.")
    if not fm.live:
        print(f"No {KEY_ENV[a.provider]} set: using cached FM results only; uncached calls will count as REVIEW.")
    df, contracts = scored_split(a.split)
    name = f"hybrid_{a.split}_{a.prompt}_{a.model.replace('/', '--')}_s{a.samples}"
    print(f"{a.split}: {contracts.contract_id.nunique()} contracts, model {a.model}, {a.prompt}-shot; calling the FM "
          f"with {a.workers} parallel workers (cached answers are free) ...", flush=True)
    live_lat, _ = prefetch(df, settings, fm, a.workers)
    if a.split == "dev":
        best, table, cost = tune(df, contracts, settings, fm)
        table.to_csv(RES / f"{name}_threshold_sweep.csv", index=False)
        settings.update(best)
        print(f"tuned on dev by business cost: {best} -> avoidable cost ${cost:.2f} per contract (config/business.json)")
    res, _ = run(df, contracts, settings, fm)
    failed = int((res.fm_used & (res.fm_status != "ok")).sum())
    if failed:
        print(f"WARNING: {failed} FM calls failed or were not cached (counted as REVIEW). See results/fm_calls.jsonl.")
        if a.split == "test":
            res.drop(columns=["out_obj"]).to_csv(RES / "hybrid_test_INVALID_contracts.csv", index=False)
            sys.exit("Test summary NOT written because some FM calls failed; fix the cause, then rerun.")
    summ = summarise(res, name); summ.update(fm_component(res, df))
    if not live_lat:                                              # everything cached: use the logged API latencies
        log = [json.loads(l) for l in open(RES / "fm_calls.jsonl")] if (RES / "fm_calls.jsonl").exists() else []
        live_lat = [r["latency_s"] for r in log if r.get("model") == a.model and not r.get("cached") and r.get("latency_s")]
    if live_lat:                                                  # real API latency, not the near-zero cache replays
        summ["latency_p50_s"], summ["latency_p95_s"] = float(np.median(live_lat)), float(np.percentile(live_lat, 95))
    summ.update({k: settings.get(k) for k in ("T_REVIEW", "FM_CONF", "T_FLAG_MIN")})
    if a.split == "test":
        h = res[res.has_ld == 1].hit5.values; rng = np.random.default_rng(SEED)
        bs = [rng.choice(h, len(h)).mean() for _ in range(2000)]
        summ["hit@5_CI95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res.drop(columns=["out_obj"]).to_csv(RES / f"{name}_contracts.csv", index=False)
    json.dump(summ, open(RES / (f"hybrid_test_summary.json" if a.split == "test" else f"{name}_summary.json"), "w"),
              indent=2, default=float)
    print(readable(summ, name))
    print(f"(full numbers: results/{'hybrid_test_summary.json' if a.split == 'test' else name + '_summary.json'})")
    if a.split == "dev" and a.freeze:
        fz = json.load(open(RES / "frozen_settings.json"))
        fz.update({"T_REVIEW_frozen": settings["T_REVIEW"], "T_REVIEW": settings["T_REVIEW"], "PROMPT": a.prompt,
                   "MODEL": a.model, "PROVIDER": a.provider, "SAMPLES": a.samples, "FM_CONF": settings["FM_CONF"],
                   "T_FLAG_MIN": settings["T_FLAG_MIN"]})
        json.dump(fz, open(RES / "frozen_settings.json", "w"), indent=2); print("frozen ->", fz)
