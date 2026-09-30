"""DEVELOPMENT ONLY: out-of-fold scores of the chunk-level model on the 408 train contracts.
Produces the numbers used to FREEZE settings before the one-shot test run:
  - hit@k for k = 1, 3, 5, 10, 20
  - contract-level threshold sweep (recall / precision / FP chunks)
  - T_BYPASS: lowest score whose chunk "touch" precision >= 0.8 (ML may FLAG without the FM)
  - candidate coverage for top-N + keyword hits (how many chunks the FM must verify)
  - T_REVIEW_ML: REVIEW threshold for ML-only mode (app without an API key)."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import SEED, K_BROAD, load_dev, fit_lr

def dev_oof(chunks, contracts):
    chunks = chunks.copy(); chunks["oof"] = np.nan; cids = contracts.contract_id.values
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(cids, contracts.has_ld, groups=cids):
        ia, ib = chunks.contract_id.isin(cids[a]), chunks.contract_id.isin(cids[b])
        v, m = fit_lr(chunks[ia].text, chunks[ia].label); chunks.loc[ib, "oof"] = m.predict_proba(v.transform(chunks[ib].text))[:, 1]
    return chunks

if __name__ == "__main__":
    chunks, contracts, spans = load_dev()
    ld = set(contracts[contracts.has_ld == 1].contract_id)
    c = dev_oof(chunks, contracts); c["touch"] = (c.ld_span_ids != "").astype(int)
    c["rank"] = c.groupby("contract_id").oof.rank(ascending=False, method="first")
    for k in (1, 3, 5, 10, 20):
        h = c[(c["rank"] <= k) & c.contract_id.isin(ld)].groupby("contract_id").label.max()
        print(f"hit@{k}: {int(h.sum())}/{len(ld)}")
    cm = c.groupby("contract_id").oof.max(); y = cm.index.isin(ld); non = c[~c.contract_id.isin(ld)]
    sweep = pd.DataFrame([{"thr": t, "c_recall": ((cm >= t) & y).sum() / y.sum(),
                           "c_prec": ((cm >= t) & y).sum() / max((cm >= t).sum(), 1),
                           "FP_chunks/nonLD": (non.oof >= t).sum() / non.contract_id.nunique(),
                           "chunk_touch_prec": c.touch[c.oof >= t].mean()} for t in np.round(np.arange(0.05, 0.96, 0.05), 2)])
    print(sweep.round(3).to_string(index=False))
    t_bypass = next((round(float(t), 2) for t in np.arange(0.5, 0.991, 0.01) if (c.oof >= t).any() and c.touch[c.oof >= t].mean() >= 0.8), 0.99)
    cov = []
    for n in (5, 10, 20, 30):
        cand = c[(c["rank"] <= n) | c.text.str.contains(K_BROAD)]
        cov.append({"N": n, "cand_per_contract": cand.groupby("contract_id").size().mean(),
                    "LD_contracts_covered": f"{cand[cand.label == 1].contract_id.nunique()}/{len(ld)}"})
    cov = pd.DataFrame(cov); print(cov.round(1).to_string(index=False)); print("T_BYPASS =", t_bypass)
    c[["contract_id", "chunk_id", "label", "bucket", "oof"]].to_csv("results/dev_oof.csv.gz", index=False)
    sweep.to_csv("results/dev_threshold_sweep.csv", index=False); cov.to_csv("results/dev_candidates.csv", index=False)
    # ML-only mode (no FM): REVIEW threshold = highest score that still surfaces >= 90% of dev LD contracts
    t_review_ml = float(sweep[sweep.c_recall >= 0.90].thr.max())
    print("T_REVIEW_ML =", t_review_ml)
    path = "results/frozen_settings.json"
    try: frozen = json.load(open(path))
    except FileNotFoundError: frozen = {}
    frozen.update({"T_BYPASS": t_bypass, "N_CAND": 20, "T_REVIEW_ML": t_review_ml})   # keeps hybrid keys if present
    json.dump(frozen, open(path, "w"), indent=2)
