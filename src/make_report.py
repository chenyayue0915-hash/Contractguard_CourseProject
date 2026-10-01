"""Build every table and figure the report needs from the files in results/ (no API calls, nothing is re-tuned).

    python src/make_report.py

Output: results/report_tables.md  and  figures/fig1_ladder_test.png, fig2_model_tradeoff.png, fig3_monthly_cost.png
Sections whose inputs do not exist yet (e.g. the hybrid test run) are listed as TODO instead of failing.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from common import K_LIT, K_BROAD, load_test
from business_case import load_business, avoidable_cost_per_contract, cost_rows, class_rates as bc_class_rates
from pipeline import load_settings
from router import decide

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
# Reference palette (light mode) from the dataviz method; text never wears a series colour.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

def pct(v): return "" if v is None or pd.isna(v) else f"{100 * v:.0f}%"
def usd(v): return "" if v is None or pd.isna(v) else (f"${v:,.4f}" if v < 0.1 else f"${v:,.2f}")

def class_rates(labels, has_ld):
    ld = [l for l, h in zip(labels, has_ld) if h]; non = [l for l, h in zip(labels, has_ld) if not h]
    return {"ld_flag": ld.count("FLAG") / len(ld), "ld_review": ld.count("REVIEW") / len(ld),
            "ld_silent": ld.count("NO_FLAG") / len(ld), "non_flag": non.count("FLAG") / len(non),
            "non_review": non.count("REVIEW") / len(non)}

def ladder_rows(cfg):
    """Keyword and ML-only systems re-labelled on the official test set with FROZEN settings (no tuning here)."""
    te, tec, _ = load_test(); s = load_settings()
    sc = pd.read_csv(RES / "test_scores.csv.gz")[["contract_id", "chunk_id", "p"]]
    te = te.merge(sc, on=["contract_id", "chunk_id"])
    has = tec.set_index("contract_id").has_ld
    prior = pd.read_csv(RES / "test_results.csv", index_col=0)
    rows = []
    systems = [("Keyword rule: \"liquidated damages\"", "keyword_literal", lambda g: int(g.text.str.contains(K_LIT).any())),
               ("Keyword rule: broad list", "keyword_broad", lambda g: int(g.text.str.contains(K_BROAD).any())),
               ("ML only (TF-IDF + LR)", "M2_chunk_level", None)]
    for name, key, kw in systems:
        labels, hs = [], []
        for cid, g in te.groupby("contract_id"):
            if kw: lab = "FLAG" if kw(g) else "NO_FLAG"
            else: lab = decide(g.p.max(), None, s["T_BYPASS"], s["T_REVIEW_ML"])[0]
            labels.append(lab); hs.append(int(has[cid]))
        c = class_rates(labels, hs)
        rows.append({"system": name, "hit@5": f"{prior.loc[key, 'hit@5']} {prior.loc[key, 'hit@5_CI95']}",
                     "hit5_rate": prior.loc[key, "hit@5_rate"], "silent_misses": f"{round(c['ld_silent'] * sum(hs))}/{sum(hs)}",
                     "flag_precision": (c["ld_flag"] * sum(hs)) / max(c["ld_flag"] * sum(hs) + c["non_flag"] * (len(hs) - sum(hs)), 1e-9),
                     "review_rate": labels.count("REVIEW") / len(labels), "fm_cost": 0.0,
                     "avoidable": avoidable_cost_per_contract(c, cfg)})
    hj = RES / "hybrid_test_summary.json"
    if hj.exists():
        h = json.loads(hj.read_text())
        c = {"ld_flag": h["flag_recall"], "ld_review": h.get("ld_review_rate", 0), "ld_silent": h.get("ld_silent_rate", 0),
             "non_flag": h["nonld_false_flag_rate"], "non_review": h["nonld_review_rate"]}
        ci = h.get("hit@5_CI95"); ci = f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
        rows.append({"system": f"Hybrid: ML + {s.get('MODEL', 'FM')} ({s.get('PROMPT', '')}-shot)", "hit@5": h["hit@5"] + ci,
                     "hit5_rate": h["hit@5_rate"], "silent_misses": h["silent_miss"], "flag_precision": h["flag_precision"],
                     "review_rate": h["abstention_rate"], "fm_cost": h["cost_per_contract_usd"],
                     "avoidable": avoidable_cost_per_contract(c, cfg) + h["cost_per_contract_usd"]})
    return pd.DataFrame(rows), hj.exists()

def md_table(df, cols, heads):
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)

def style(ax, title, subtitle):
    ax.set_facecolor(SURFACE); ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.figure.text(0.02, 0.965, title, fontsize=12, fontweight="bold", color=INK, va="top")
    ax.figure.text(0.02, 0.905, subtitle, fontsize=9, color=INK2, va="top")

def fig_ladder(lad):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = lad.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(d) + 1.6)); fig.subplots_adjust(left=0.36, right=0.93, top=0.78, bottom=0.14)
    ax.barh(d.system, d.hit5_rate, color=SERIES[0], height=0.55)
    for y, (v, lab) in enumerate(zip(d.hit5_rate, d["hit@5"])):
        ax.text(v + 0.01, y, lab.split(" ")[0], va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("LD contracts with a real LD passage among the 5 shown (hit@5)", color=INK2, fontsize=9)
    top = lad.sort_values("hit5_rate").iloc[-1]; kw = lad[lad.system.str.startswith("Keyword")].hit5_rate.max()
    style(ax, f"{top.system.split(' (')[0]} finds {top['hit@5'].split(' ')[0]} LD contracts; keyword rules {round(kw * 14)}/14",
          "Official CUAD test set, 102 contracts (14 with LD), scored once")
    FIG.mkdir(exist_ok=True); fig.savefig(FIG / "fig1_ladder_test.png", dpi=200); plt.close(fig)

def fig_tradeoff(mc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = mc.dropna(subset=["avoidable_cost_per_contract"])
    if len(d) < 2: return False
    fig, ax = plt.subplots(figsize=(8, 4.6)); fig.subplots_adjust(left=0.12, right=0.95, top=0.8, bottom=0.14)
    x = d.cost_per_contract_usd * 1000; y = d.avoidable_cost_per_contract
    base = d.model.eq("ML only (no FM)")
    ax.scatter(x[~base], y[~base], s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter(x[base], y[base], s=64, color=INK2, edgecolor=SURFACE, linewidth=2, zorder=3)
    xmax = max(x.max(), 1e-9)
    for xi, yi, lab in zip(x, y, d.label + np.where(d.prompt.astype(str).str.len() > 0, " · " + d.prompt.astype(str), "")):
        right = xi > 0.7 * xmax
        ax.annotate(lab, (xi, yi), xytext=(-6 if right else 6, 5), textcoords="offset points", fontsize=8.5, color=INK,
                    ha="right" if right else "left")
    ax.set_xlim(-0.05 * xmax, 1.1 * xmax); ax.set_ylim(0, y.max() * 1.15)
    ax.set_xlabel("FM spend per 1,000 contracts (USD, billed by OpenRouter)", color=INK2, fontsize=9)
    ax.set_ylabel("Avoidable cost per contract (USD)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    b = d.sort_values("avoidable_cost_per_contract").iloc[0]
    style(ax, f"Lowest avoidable cost: {b.label} at ${b.avoidable_cost_per_contract:.2f} per contract",
          "Development subset (107 train contracts); lower is better on both axes")
    fig.savefig(FIG / "fig2_model_tradeoff.png", dpi=200); plt.close(fig); return True

def fig_monthly(bc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = bc.iloc[::-1]; parts = [("fm_spend", "FM spend"), ("review_time_cost", "REVIEW handling (time + escalations)"),
                                ("unneeded_lawyer_cost", "Unneeded lawyer consults"), ("expected_loss_missed", "Expected loss from missed LD")]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(d) + 2.0)); fig.subplots_adjust(left=0.3, right=0.95, top=0.74, bottom=0.2)
    left = np.zeros(len(d))
    for (col, name), colr in zip(parts, SERIES):
        ax.barh(d.system, d[col], left=left, color=colr, height=0.55, label=name, edgecolor=SURFACE, linewidth=1.5)
        left += d[col].values
    for y, v in enumerate(left): ax.text(v, y, f"  ${v:,.0f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, left.max() * 1.18); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    best = bc.sort_values("total_avoidable_cost").iloc[0]
    style(ax, f"{best.system} has the lowest avoidable cost: ${best.total_avoidable_cost:,.0f} a month",
          "4,000 contracts a month; assumptions in config/business.json")
    fig.savefig(FIG / "fig3_monthly_cost.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    cfg = load_business(); s = load_settings(); out = ["# ContractGuard: numbers for the report", "",
          "Generated by `python src/make_report.py` from the files in `results/`. Do not edit by hand.", ""]
    lad, has_hybrid = ladder_rows(cfg)
    lad_md = lad.assign(flag_precision=lad.flag_precision.map(pct), review_rate=lad.review_rate.map(pct),
                        fm_cost=lad.fm_cost.map(usd), avoidable=lad.avoidable.map(usd))
    out += ["## 1. Technique ladder (official test set, 102 contracts, 14 with LD)", "",
            md_table(lad_md, ["system", "hit@5", "silent_misses", "flag_precision", "review_rate", "fm_cost", "avoidable"],
                     ["system", "hit@5 [95% CI]", "LD contracts missed silently", "FLAG precision", "sent to a person",
                      "FM cost / contract", "avoidable cost / contract"]), ""]
    if not has_hybrid: out += ["TODO: hybrid row appears after `python src/evaluate_hybrid.py --split test`.", ""]
    fig_ladder(lad)
    mcp = RES / "model_comparison.csv"
    if mcp.exists():
        mc = pd.read_csv(mcp); mc["prompt"] = mc.get("prompt", pd.Series([""] * len(mc))).fillna("")
        # One consistent business figure for every row (ML-only included): class rates re-weighted to real prevalence
        mc["avoidable_cost_per_contract"] = [avoidable_cost_per_contract(bc_class_rates(r), cfg) + r["cost_per_contract_usd"]
                                             for _, r in mc.iterrows()]
        view = mc.assign(flag_precision=mc.flag_precision.map(pct), abstention_rate=mc.abstention_rate.map(pct),
                         cost=mc.cost_per_contract_usd.map(usd),
                         price=mc.apply(lambda r: "" if pd.isna(r.get("openrouter_in")) else f"${r['openrouter_in']:.2f} / ${r['openrouter_out']:.2f}", axis=1),
                         latency=mc.apply(lambda r: "" if pd.isna(r.get("latency_p50_s")) else f"{r['latency_p50_s']:.1f} / {r['latency_p95_s']:.1f} s", axis=1),
                         avoid=mc.avoidable_cost_per_contract.map(usd),
                         failures=mc.get("fm_call_failures", pd.Series([""] * len(mc))).fillna(""))
        out += ["## 2. Which FM to rent (development subset: 107 train contracts, 47 with LD)", "",
                md_table(view, ["label", "prompt", "price", "hit@5", "silent_miss", "flag_precision", "abstention_rate",
                                "failures", "cost", "latency", "avoid"],
                         ["model", "prompt", "$ per 1M tokens in / out", "hit@5", "missed silently", "FLAG precision",
                          "sent to a person", "failed FM calls", "FM cost / contract", "latency p50 / p95", "avoidable cost / contract"]), ""]
        pairs = mc[mc.prompt.isin(["zero", "few"])].groupby("model").filter(lambda g: g.prompt.nunique() == 2)
        out += ["## 3. Prompt techniques (Class 3)", ""]
        if len(pairs):
            out += [md_table(pairs.assign(cost=pairs.cost_per_contract_usd.map(usd), fp=pairs.flag_precision.map(pct)),
                             ["label", "prompt", "hit@5", "silent_miss", "fp", "cost"],
                             ["model", "prompt", "hit@5", "missed silently", "FLAG precision", "FM cost / contract"]), ""]
        else:
            out += ["TODO: run `python src/compare_models.py --only <model> --prompt few` for a model already run zero-shot.", ""]
        cheap = mc[mc.cost_per_contract_usd > 0]
        if len(cheap):
            r = cheap.sort_values("cost_per_contract_usd").iloc[0]
            out += [f"Self-consistency (3 samples, majority vote) was not run; it triples the FM calls, so for {r['label']} the FM cost "
                    f"would rise from {usd(r['cost_per_contract_usd'])} to about {usd(3 * r['cost_per_contract_usd'])} per contract "
                    "(and latency roughly triples if the samples run one after another).", ""]
        fig_tradeoff(mc)
        bc = cost_rows(mc, cfg, cfg["contracts_per_month"]["value"])
        out += ["## 4. Business case: cost to serve (Class 5)", "", (RES / "business_case.md").read_text() if (RES / "business_case.md").exists()
                else "TODO: run `python src/business_case.py`.", ""]
        fig_monthly(bc)
    else:
        out += ["TODO: run `python src/compare_models.py` first (sections 2-4).", ""]
    stress = [p for p in (RES / "stress_summary.csv", RES / "stress_summary_fm.csv") if p.exists()]
    if stress:
        st = pd.read_csv(stress[-1])
        piv = st.pivot_table(index="system", columns="type", values=["FLAG", "NO_FLAG"])
        rows = []
        for sysn in piv.index:
            rows.append({"system": sysn, "para": pct(piv.loc[sysn, ("NO_FLAG", "paraphrase_ld")]),
                         "inj": pct(piv.loc[sysn, ("NO_FLAG", "injection_ld")]), "hard": pct(piv.loc[sysn, ("FLAG", "hard_negative")])})
        out += ["## 5. Stress test (50 clauses, labels fixed before running)", "",
                md_table(pd.DataFrame(rows), ["system", "para", "inj", "hard"],
                         ["system", "paraphrased LD missed (NO_FLAG)", "injected LD missed (NO_FLAG)", "hard negatives flagged"]), ""]
        if len(stress) == 1: out += ["TODO: hybrid rows appear after `python src/stress_eval.py --fm`.", ""]
    model = s.get("MODEL", "(not frozen yet)")
    out += ["## 6. Build vs buy (Class 2 stack)", "",
            "| layer | own / rent | what | why |", "|---|---|---|---|",
            "| interface & serving | rent | Streamlit (open source) | a one-page upload-and-read UI is commodity |",
            "| orchestration | own | `src/pipeline.py`, `src/router.py` | the abstention and safety rules are the product and must be auditable |",
            "| model: retrieval | own | TF-IDF + logistic regression (scikit-learn) | labels exist; CPU, milliseconds, ~$0 per contract |",
            f"| model: verification | rent | {model} via OpenRouter | fine-tuning or hosting a model is not justified by 121 positive examples |",
            "| data | rent + own | CUAD v1 (CC BY 4.0) + own chunk-level labelling | dataset is public; the negative design is ours |",
            "| evaluation & observability | own | `evaluate_hybrid.py`, `compare_models.py`, `fm_calls.jsonl` | metrics are specific to this problem |", "",
            "## Figures", "", "- `figures/fig1_ladder_test.png`", "- `figures/fig2_model_tradeoff.png`", "- `figures/fig3_monthly_cost.png`", ""]
    (RES / "report_tables.md").write_text("\n".join(out)); print("\n".join(out))
