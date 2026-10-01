"""Cost-to-serve and business trade-off (course Class 5): turns the measured rates of each system into money per month.

    python src/business_case.py                      # uses config/business.json
    python src/business_case.py --contracts-per-month 1000

Inputs : results/model_comparison.csv (rates measured on the development subset) + config/business.json (assumptions)
Output : results/business_case.md / .csv

Why class-conditional rates: the dev subset is enriched with LD contracts (47 of 107 = 44%), while real contracts contain
LD far less often (CUAD: 12%). Rates are therefore measured separately on LD and non-LD contracts and re-weighted with the
real prevalence, instead of reusing precision numbers that depend on the subset's mix.
"""
import argparse, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

def frac(s):
    a, b = str(s).split("/"); return int(a) / max(int(b), 1)

def class_rates(r):
    """Per-class label rates. Uses logged columns when present, otherwise derives them exactly from the aggregate
    numbers and the subset size (107 contracts, 47 with LD, for rows written before those columns existed)."""
    n = r.get("n_contracts") if pd.notna(r.get("n_contracts", float("nan"))) else 107
    n_ld = r.get("n_ld") if pd.notna(r.get("n_ld", float("nan"))) else 47
    n_non = n - n_ld
    if pd.notna(r.get("nonld_false_flag_rate", float("nan"))):
        return {"ld_flag": r["flag_recall"], "ld_review": r["ld_review_rate"], "ld_silent": r["ld_silent_rate"],
                "non_flag": r["nonld_false_flag_rate"], "non_review": r["nonld_review_rate"]}
    ld_flag = r["flag_recall"]; ld_silent = frac(r["silent_miss"]); ld_review = max(0.0, 1 - ld_flag - ld_silent)
    true_flags = ld_flag * n_ld
    all_flags = true_flags / r["flag_precision"] if r["flag_precision"] > 0 else true_flags
    non_flag = (all_flags - true_flags) / n_non
    non_review = max(0.0, (r["abstention_rate"] * n - ld_review * n_ld) / n_non)
    return {"ld_flag": ld_flag, "ld_review": ld_review, "ld_silent": ld_silent, "non_flag": non_flag, "non_review": non_review}

def load_business(path=ROOT / "config" / "business.json"):
    return json.loads(Path(path).read_text())

def review_unit_cost(cfg):
    """One REVIEW = the owner's reading time + the chance that she still pays a lawyer because she is unsure."""
    esc = cfg.get("review_escalation_rate", {"value": 0})["value"]
    return cfg["review_minutes_per_contract"]["value"] / 60 * cfg["reviewer_cost_per_hour"]["value"] \
        + esc * cfg["lawyer_consult_cost"]["value"]

def avoidable_cost_per_contract(c, cfg):
    """c = class rates {ld_flag, ld_review, ld_silent, non_flag, non_review}; FM spend is added by the caller."""
    prev = cfg["ld_prevalence"]["value"]
    review = (prev * c["ld_review"] + (1 - prev) * c["non_review"]) * review_unit_cost(cfg)
    return review + (1 - prev) * c["non_flag"] * cfg["lawyer_consult_cost"]["value"] \
        + prev * c["ld_silent"] * cfg["loss_per_missed_ld_clause"]["value"]

def unusable(r):
    """True when every FM call failed (e.g. a model that cannot do forced tool calls): its rates only show the fallback."""
    f = r.get("fm_call_failures")
    if not isinstance(f, str) or "/" not in f: return False
    a, b = f.split("/"); return int(b) > 0 and int(a) == int(b)

def cost_rows(df, cfg, n_month):
    df = df[[not unusable(r) for _, r in df.iterrows()]]
    prev = cfg["ld_prevalence"]["value"]; mins = cfg["review_minutes_per_contract"]["value"]
    hourly = cfg["reviewer_cost_per_hour"]["value"]; lawyer = cfg["lawyer_consult_cost"]["value"]
    loss = cfg["loss_per_missed_ld_clause"]["value"]
    ld, non = n_month * prev, n_month * (1 - prev); rows = []
    for _, r in df.iterrows():
        c = class_rates(r)
        flags_true, flags_false = ld * c["ld_flag"], non * c["non_flag"]
        reviews = ld * c["ld_review"] + non * c["non_review"]; missed = ld * c["ld_silent"]
        fm = r["cost_per_contract_usd"] * n_month
        rev_cost = reviews * review_unit_cost(cfg); false_cost = flags_false * lawyer; miss_cost = missed * loss
        pr = r.get("prompt") if isinstance(r.get("prompt"), str) else ""
        rows.append({"system": r["label"] + ((f" ({pr}-shot)" if pr in ("zero", "few") else f" ({pr})") if pr else ""),
                     "flags_needed": flags_true, "flags_unneeded": flags_false, "reviews": reviews, "missed_ld": missed,
                     "fm_spend": fm, "review_time_cost": rev_cost, "unneeded_lawyer_cost": false_cost,
                     "expected_loss_missed": miss_cost, "total_avoidable_cost": fm + rev_cost + false_cost + miss_cost})
    out = pd.DataFrame(rows); out["per_contract"] = out.total_avoidable_cost / n_month
    return out.sort_values("total_avoidable_cost").reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--contracts-per-month", type=int); a = ap.parse_args()
    cfg = json.loads((ROOT / "config" / "business.json").read_text())
    n = a.contracts_per_month or cfg["contracts_per_month"]["value"]
    df = pd.read_csv(RES / "model_comparison.csv")
    out = cost_rows(df, cfg, n); out.to_csv(RES / "business_case.csv", index=False)
    m = lambda v: f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"
    lines = [f"Monthly cost of each design for {n:,} contracts ({cfg['currency']}). LD prevalence "
             f"{cfg['ld_prevalence']['value']:.0%}; rates measured on the development subset, re-weighted to that prevalence.", "",
             "| system | FM spend | REVIEW handling (owner time + escalations) | unneeded lawyer consults on FLAG | expected loss from missed LD | total avoidable cost | per contract |",
             "|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r.system} | {m(r.fm_spend)} | {m(r.review_time_cost)} ({r.reviews:,.0f} reviews) | "
                     f"{m(r.unneeded_lawyer_cost)} ({r.flags_unneeded:,.0f}) | {m(r.expected_loss_missed)} ({r.missed_ld:,.1f} missed) | "
                     f"**{m(r.total_avoidable_cost)}** | {m(r.per_contract)} |")
    # Sensitivity: does the ranking survive other values of the most uncertain assumption?
    lines += ["", "Sensitivity to the loss caused by one missed LD clause (total avoidable cost per month):", "",
              "| system | " + " | ".join(f"loss ${v:,}" for v in (1000, 5000, 20000)) + " |", "|---|---|---|---|"]
    for sysname in out.system:
        vals = []
        for v in (1000, 5000, 20000):
            c2 = json.loads(json.dumps(cfg)); c2["loss_per_missed_ld_clause"]["value"] = v
            vals.append(cost_rows(df, c2, n).set_index("system").loc[sysname, "total_avoidable_cost"])
        lines.append(f"| {sysname} | " + " | ".join(m(x) for x in vals) + " |")
    # Sensitivity: the hybrids never miss, so their ranking hinges on what one REVIEW really costs.
    lines += ["", "Sensitivity to the review escalation rate (share of REVIEW cases that still end in a paid lawyer consult):", "",
              "| system | " + " | ".join(f"escalation {v:.0%}" for v in (0.0, 0.25, 0.5)) + " |", "|---|---|---|---|"]
    for sysname in out.system:
        vals = []
        for v in (0.0, 0.25, 0.5):
            c2 = json.loads(json.dumps(cfg)); c2.setdefault("review_escalation_rate", {"value": 0, "source": ""})["value"] = v
            vals.append(cost_rows(df, c2, n).set_index("system").loc[sysname, "total_avoidable_cost"])
        lines.append(f"| {sysname} | " + " | ".join(m(x) for x in vals) + " |")
    unit = review_unit_cost(cfg); one_pp = n * 0.01 * unit
    lines += ["", f"One REVIEW costs {m(unit)}, so cutting the REVIEW rate by one percentage point saves about {m(one_pp)} a month; "
              "compare that with the FM spend column when judging a more expensive verifier."]
    bad = [r["label"] for _, r in df.iterrows() if unusable(r)]
    if bad:
        lines += ["", "Left out because every FM call failed (the model could not be used with forced tool calling): " + ", ".join(bad) + "."]
    lines += ["", "Lawyer consults for contracts that really contain LD are needed in every design and are left out.", "",
              "Assumptions (config/business.json):", ""]
    lines += [f"- {k}: {v['value']} — {v['source']}" for k, v in cfg.items() if isinstance(v, dict)]
    (RES / "business_case.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
