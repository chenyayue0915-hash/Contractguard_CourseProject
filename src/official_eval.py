"""FINAL, RUN ONCE. Settings are frozen on the 408 train contracts (oof_analysis.py -> results/frozen_settings.json),
then the 102 official test contracts are scored a single time. Also trains and saves the deployable model."""
import json, random, numpy as np, pandas as pd, joblib
from common import SEED, K_LIT, K_BROAD, MASK, load_dev, load_test, load_cuad_docs, fit_lr, m1_train_texts, metrics

if __name__ == "__main__":
    frozen = json.load(open("results/frozen_settings.json"))          # produced on dev only
    T_BYPASS, N_CAND = frozen["T_BYPASS"], frozen["N_CAND"]
    tr, trc, _ = load_dev(); te, tec, tsp = load_test()
    ld_ids = set(trc[trc.has_ld == 1].contract_id) | set(tec[tec.has_ld == 1].contract_id); n_sp = len(tsp)
    rows, hits = {}, {}
    rows["keyword_literal"], hits["keyword_literal"] = metrics(te, te.text.str.count(K_LIT), ld_ids, n_sp, 1)
    rows["keyword_broad"], hits["keyword_broad"] = metrics(te, te.text.str.count(K_BROAD), ld_ids, n_sp, 1)
    docs = load_cuad_docs(keep=set(trc.contract_id))
    X, y = m1_train_texts(docs, set(trc.contract_id), random.Random(SEED)); v, m = fit_lr(X, y)
    rows["M1_span_negatives"], hits["M1_span_negatives"] = metrics(te, m.predict_proba(v.transform(te.text))[:, 1], ld_ids, n_sp, 0.5)
    v, m = fit_lr(tr.text, tr.label); te["p"] = m.predict_proba(v.transform(te.text))[:, 1]
    rows["M2_chunk_level"], hits["M2_chunk_level"] = metrics(te, te.p, ld_ids, n_sp, 0.5)
    joblib.dump({"vectorizer": v, "model": m, "T_BYPASS": T_BYPASS, "N_CAND": N_CAND}, "models/tfidf_lr.joblib")
    vm, mm = fit_lr(tr.text.str.replace(MASK, " ", regex=True), tr.label)
    rows["M2_masked"], hits["M2_masked"] = metrics(te, mm.predict_proba(vm.transform(te.text.str.replace(MASK, " ", regex=True)))[:, 1], ld_ids, n_sp, 0.5)
    rng = np.random.default_rng(SEED)
    for k, h in hits.items():
        bs = [rng.choice(h.values, len(h)).mean() for _ in range(2000)]
        rows[k]["hit@5_CI95"] = f"[{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"
    res = pd.DataFrame(rows).T; pd.set_option("display.width", 220)
    print(f"TEST: {len(tec)} contracts, {tec.has_ld.sum()} with LD, {n_sp} spans\n", res); res.to_csv("results/test_results.csv")
    te["rank"] = te.groupby("contract_id").p.rank(ascending=False, method="first")
    cand = te[(te["rank"] <= N_CAND) | te.text.str.contains(K_BROAD)]
    touch = (te.ld_span_ids != "").astype(int)
    print(f"candidates/contract {cand.groupby('contract_id').size().mean():.1f}; LD contracts covered "
          f"{cand[cand.label == 1].contract_id.nunique()}/{tec.has_ld.sum()}; ML-bypass chunks {(te.p >= T_BYPASS).sum()} "
          f"(touch precision {touch[te.p >= T_BYPASS].mean():.2f})")
    print("M2 missed:", hits["M2_chunk_level"][hits["M2_chunk_level"] == 0].index.tolist())
    te[["contract_id", "chunk_id", "label", "bucket", "p"]].to_csv("results/test_scores.csv.gz", index=False)
