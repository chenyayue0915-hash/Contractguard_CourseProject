"""Shared loaders, the official train/test split, and metrics.

Split rule (enforced here, not by convention):
  * Development scripts (experiment_cv.py, oof_analysis.py) call `load_dev()`,
    which returns ONLY the 408 official train contracts and asserts no test contract leaks in.
  * Only official_eval.py / evaluate_hybrid.py may call `load_test()`, once, with settings frozen on dev.
"""
import json, re, random
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 42
PROC, RAW = "data/processed", "data/raw"
LD = "Liquidated Damages"

# Keyword lists are fixed here BEFORE any test run (never edited after looking at errors).
K_LIT = re.compile(r"liquidated\s+damages", re.I)
K_BROAD = re.compile(r"liquidated|penalt|per (?:calendar )?day|pre-estimate|termination fee|break-?up fee|exit fee|lump[- ]sum", re.I)
MASK = re.compile(r"liquidated\s+damages?|penalt(?:y|ies)", re.I)   # leakage check: hide the literal label words

def test_ids():
    return set(json.load(open(f"{RAW}/test_titles.json")))

def _load():
    chunks = pd.read_csv(f"{PROC}/chunks.csv.gz"); chunks["ld_span_ids"] = chunks.ld_span_ids.fillna("").astype(str)
    contracts = pd.read_csv(f"{PROC}/contracts.csv"); spans = pd.read_csv(f"{PROC}/ld_spans.csv")
    return chunks, contracts, spans

def _subset(ids, *dfs):
    return [d[d.contract_id.isin(ids)].reset_index(drop=True) for d in dfs]

def assert_no_test(*dfs):
    t = test_ids()
    for d in dfs:
        leaked = set(d.contract_id) & t
        assert not leaked, f"{len(leaked)} TEST contracts found in a development dataframe"

def load_dev():
    """408 official train contracts only."""
    chunks, contracts, spans = _load()
    ids = set(contracts.contract_id) - test_ids()
    out = _subset(ids, chunks, contracts, spans); assert_no_test(*out)
    return out

def load_test():
    """102 official test contracts. Call only from the one-shot final evaluation."""
    chunks, contracts, spans = _load()
    return _subset(test_ids(), chunks, contracts, spans)

def load_cuad_docs(path=f"{RAW}/CUADv1.json", keep=None):
    d = json.load(open(path)); docs = {}
    for doc in d["data"]:
        if keep is not None and doc["title"] not in keep: continue
        p = doc["paragraphs"][0]; spans = {}
        for qa in p["qas"]:
            spans[qa["id"].split("__")[-1]] = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
        docs[doc["title"]] = {"text": p["context"], "spans": spans}
    return docs

def vec():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200_000, stop_words="english")

def fit_lr(texts, y):
    v = vec(); m = LogisticRegression(class_weight="balanced", max_iter=2000, C=4).fit(v.fit_transform(texts), y)
    return v, m

def m1_train_texts(docs, train_ids, rng):
    """Milestone-1 design (kept only as a comparison): LD spans vs other-category spans at 1:3."""
    pos, neg = [], []
    for cid in sorted(train_ids):
        d = docs[cid]; ld = d["spans"].get(LD, [])
        pos += [d["text"][a:b] for a, b in ld]
        for c, ss in d["spans"].items():
            if c == LD: continue
            neg += [d["text"][a:b] for a, b in ss if not any(a < y and b > x for x, y in ld)]
    neg = rng.sample(neg, min(len(neg), 3 * len(pos)))
    return pos + neg, [1] * len(pos) + [0] * len(neg)

def metrics(df, score, ld_ids, n_spans, thr=None, k=5):
    """hit@k (headline), span Recall@k, and contract-level recall/precision + FP chunks at a threshold."""
    df = df.assign(s=np.asarray(score)); out = {}
    top = df.sort_values("s", ascending=False).groupby("contract_id").head(k)
    ldc = [c for c in df.contract_id.unique() if c in ld_ids]
    hit = top[top.contract_id.isin(ld_ids)].groupby("contract_id").label.max().reindex(ldc, fill_value=0)
    out[f"hit@{k}"] = f"{int(hit.sum())}/{len(ldc)}"; out[f"hit@{k}_rate"] = hit.mean()
    found = {(r.contract_id, j) for r in top.itertuples() for j in r.ld_span_ids.split(",") if j}
    out[f"span_R@{k}"] = len(found) / n_spans
    if thr is not None:
        cm = df.groupby("contract_id").s.max(); y = cm.index.isin(ld_ids); f = (cm >= thr).values
        out["c_recall"] = (f & y).sum() / y.sum(); out["c_prec"] = (f & y).sum() / max(f.sum(), 1)
        non = df[~df.contract_id.isin(ld_ids)]; out["FP_chunks/nonLD"] = (non.s >= thr).sum() / non.contract_id.nunique()
    return out, hit
