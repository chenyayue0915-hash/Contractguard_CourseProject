"""End-to-end analysis of one contract. Used by the Streamlit app AND by evaluate_hybrid.py,
so the evaluated system is exactly the deployed system."""
import json
from pathlib import Path
import joblib
from chunker import chunk_text
from common import K_BROAD
from router import decide, rank

ROOT = Path(__file__).resolve().parent.parent
MAX_CAND, K_SHOW = 30, 5

def load_settings():
    s = json.load(open(ROOT / "results" / "frozen_settings.json"))
    s.setdefault("T_REVIEW_ML", 0.10)            # written by oof_analysis.py (dev only)
    s.setdefault("T_REVIEW", 0.30); s.setdefault("FM_CONF", 0.5)   # T_REVIEW is tuned by evaluate_hybrid.py --split dev
    return s

def candidates(texts, probs, n_cand):
    """Retrieval step: ML top-N plus every keyword hit, capped at MAX_CAND (highest ML score first)."""
    order = sorted(range(len(texts)), key=lambda i: -probs[i])
    chosen = set(order[:n_cand]) | {i for i, t in enumerate(texts) if K_BROAD.search(t)}
    return sorted(chosen, key=lambda i: -probs[i])[:MAX_CAND]

def analyze_scored(texts, probs, settings, fm_verifier=None, tag=""):
    """texts/probs: one entry per chunk (already scored)."""
    max_p = max(probs) if probs else 0.0
    cand = candidates(texts, probs, settings["N_CAND"])
    fm = None
    if fm_verifier is not None and max_p < settings["T_BYPASS"]:       # rule 1 bypasses the FM entirely
        fm = fm_verifier.verify([(i, texts[i]) for i in cand], tag=tag)
    t_review = settings["T_REVIEW"] if fm_verifier is not None else settings["T_REVIEW_ML"]
    label, reason = decide(max_p, fm, settings["T_BYPASS"], t_review, settings["FM_CONF"])
    order = rank(len(texts), probs, fm, settings["FM_CONF"])
    fr = fm["results"] if fm else {}
    top = [{"chunk": i, "p": probs[i], "fm_label": fr.get(i, {}).get("label"),
            "fm_conf": fr.get(i, {}).get("confidence"), "evidence": fr.get(i, {}).get("evidence", ""),
            "evidence_ok": fr.get(i, {}).get("valid")} for i in order[:K_SHOW]]
    return {"label": label, "reason": reason, "max_p": max_p, "top": top, "n_chunks": len(texts),
            "candidates": cand, "fm_used": fm is not None, "fm_raw": fm, "fm_status": fm["status"] if fm else None,
            "cost_usd": fm["cost_usd"] if fm else 0.0, "latency_s": fm["latency_s"] if fm else 0.0}

class ContractGuard:
    def __init__(self, model_path=ROOT / "models" / "tfidf_lr.joblib"):
        b = joblib.load(model_path); self.vec, self.model = b["vectorizer"], b["model"]
        self.settings = load_settings()

    def score(self, texts):
        return self.model.predict_proba(self.vec.transform(texts))[:, 1].tolist() if texts else []

    def analyze_text(self, text, fm_verifier=None):
        chunks = chunk_text(text); texts = [c[2] for c in chunks]
        out = analyze_scored(texts, self.score(texts), self.settings, fm_verifier)
        for t in out["top"]:
            a, b, s = chunks[t["chunk"]]; t.update(start=a, end=b, text=s)
        return out
