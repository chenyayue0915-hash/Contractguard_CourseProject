"""Stress test on eval/stress_cases.csv (labels fixed before running). Each clause is scored as a one-passage contract.
Systems: keyword rule, ML only, and the hybrid (only if an API key or cached FM results are available).
    python src/stress_eval.py            # ML-only + keyword (no API key needed)
    python src/stress_eval.py --fm       # adds the hybrid with the frozen prompt/model
"""
import argparse, json, random
from pathlib import Path
import pandas as pd
from common import K_BROAD, SEED
from pipeline import ContractGuard
from router import decide

RES = Path("results")

def fm_by_case(cases, fm, batch=10):
    """Mixed batches of 10 (injections share a call with clean clauses, as they would inside a real contract)."""
    idx = list(range(len(cases))); random.Random(SEED).shuffle(idx); out = {}
    for b in range(0, len(idx), batch):
        ids = idx[b:b + batch]
        r = fm.verify([(i, cases.text[i]) for i in ids], tag=f"stress-batch-{b // batch}")
        for i in ids:
            out[i] = {"status": "ok" if i in r["results"] else r["status"],
                      "results": {i: r["results"][i]} if i in r["results"] else {}, "cost_usd": r["cost_usd"] / len(ids)}
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fm", action="store_true"); a = ap.parse_args()
    cases = pd.read_csv("eval/stress_cases.csv")
    cg = ContractGuard(); s = cg.settings
    cases["p"] = cg.score(cases.text.tolist())
    cases["keyword"] = cases.text.str.contains(K_BROAD).map({True: "FLAG", False: "NO_FLAG"})
    cases["ml_only"] = [decide(p, None, s["T_BYPASS"], s["T_REVIEW_ML"])[0] for p in cases.p]
    systems = ["keyword", "ml_only"]
    if a.fm:
        from fm_verify import FMVerifier
        from config import load_models
        fm = FMVerifier(model=s.get("MODEL", load_models()["default_model"]), prompt=s.get("PROMPT", "few"),
                        samples=s.get("SAMPLES", 1), provider=s.get("PROVIDER", "openrouter"),
                        cache_path=RES / "fm_cache.jsonl", log_path=RES / "fm_calls.jsonl")
        per = fm_by_case(cases, fm)
        cases["hybrid"] = [decide(p, per[i], s["T_BYPASS"], s.get("T_REVIEW", 0.3), s["FM_CONF"], s.get("T_FLAG_MIN", 0.0))[0]
                           for i, p in enumerate(cases.p)]
        cases["fm_label"] = [next(iter(per[i]["results"].values()), {}).get("label") for i in range(len(cases))]
        systems.append("hybrid")
    rows = []
    for sys_ in systems:
        for t, g in cases.groupby("type"):
            rows.append({"system": sys_, "type": t, "n": len(g), "FLAG": (g[sys_] == "FLAG").mean(),
                         "REVIEW": (g[sys_] == "REVIEW").mean(), "NO_FLAG": (g[sys_] == "NO_FLAG").mean()})
    summ = pd.DataFrame(rows); pd.set_option("display.width", 200)
    print(summ.round(2).to_string(index=False))
    print("\nReading: for paraphrase_ld and injection_ld, NO_FLAG is a silent miss; for hard_negative, FLAG is a false alarm.")
    summ.to_csv(RES / f"stress_summary{'_fm' if a.fm else ''}.csv", index=False)
    cases.to_csv(RES / f"stress_cases_scored{'_fm' if a.fm else ''}.csv", index=False)
