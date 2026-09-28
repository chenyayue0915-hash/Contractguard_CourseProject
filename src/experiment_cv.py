"""DEVELOPMENT ONLY: 5-fold contract-grouped CV on the 408 official train contracts.
Compares keyword rules, the Milestone-1 design (span negatives), chunk-level training, and the masked (leakage) variant.
The 102 test contracts are never loaded here."""
import random, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import SEED, K_LIT, K_BROAD, MASK, load_dev, load_cuad_docs, fit_lr, m1_train_texts, metrics

if __name__ == "__main__":
    chunks, contracts, spans = load_dev()
    ld_ids = set(contracts[contracts.has_ld == 1].contract_id)
    docs = load_cuad_docs(keep=set(contracts.contract_id))
    cids = contracts.contract_id.values; rng = random.Random(SEED); rows = []
    for fold, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(cids, contracts.has_ld, groups=cids)):
        trc, tec = set(cids[a]), set(cids[b])
        ctr, cte = chunks[chunks.contract_id.isin(trc)], chunks[chunks.contract_id.isin(tec)]
        n_sp = spans.contract_id.isin(tec).sum(); res = {}
        res["keyword_literal"], _ = metrics(cte, cte.text.str.count(K_LIT), ld_ids, n_sp, thr=1)
        res["keyword_broad"], _ = metrics(cte, cte.text.str.count(K_BROAD), ld_ids, n_sp, thr=1)
        X, y = m1_train_texts(docs, trc, rng); v, m = fit_lr(X, y)
        res["M1_span_negatives"], _ = metrics(cte, m.predict_proba(v.transform(cte.text))[:, 1], ld_ids, n_sp, thr=0.5)
        v, m = fit_lr(ctr.text, ctr.label)
        res["M2_chunk_level"], _ = metrics(cte, m.predict_proba(v.transform(cte.text))[:, 1], ld_ids, n_sp, thr=0.5)
        v, m = fit_lr(ctr.text.str.replace(MASK, " ", regex=True), ctr.label)
        res["M2_masked"], _ = metrics(cte, m.predict_proba(v.transform(cte.text.str.replace(MASK, " ", regex=True)))[:, 1], ld_ids, n_sp, thr=0.5)
        for name, r in res.items():
            r.update(model=name, fold=fold, ld_contracts=len(tec & ld_ids), spans=n_sp); rows.append(r)
        print(f"fold {fold}: {len(tec)} contracts, {len(tec & ld_ids)} with LD, {n_sp} spans", flush=True)
    df = pd.DataFrame(rows).drop(columns=["hit@5"])
    df.to_csv("results/dev_cv_folds.csv", index=False)
    num = ["hit@5_rate", "span_R@5", "c_recall", "c_prec", "FP_chunks/nonLD"]
    summ = df.groupby("model")[num].agg(["mean", "std"]).round(3)
    summ.to_csv("results/dev_cv_summary.csv"); pd.set_option("display.width", 220); print(summ)
