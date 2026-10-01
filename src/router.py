"""Decision rules: deterministic code, not the FM, decides the contract-level label.

Order matters and is part of the safety design:
  1. ML score >= T_BYPASS            -> FLAG   (the FM is not consulted, so injected text cannot clear it)
  2. ML-only mode (no FM configured) -> REVIEW if max ML >= T_REVIEW else NO_FLAG
  3. FM call failed / incomplete     -> REVIEW (never silently NO_FLAG)
  4. FM says LD, evidence verified, confidence >= FM_CONF -> FLAG, but only if max ML >= T_FLAG_MIN
     (an FM-only "LD" on a passage the ML model found unremarkable goes to REVIEW: cheaper than a lawyer consult)
  5. FM says LD but evidence not verified or confidence low -> REVIEW
  6. max ML >= T_REVIEW but FM says no -> REVIEW (ML and FM disagree)
  7. otherwise                        -> NO_FLAG ("nothing found", which is not "safe")
"""
LABELS = ("FLAG", "REVIEW", "NO_FLAG")
REASONS = {
    "ml_high": "The ML model scored a passage very high.",
    "fm_confirmed": "The FM confirmed a passage and quoted it exactly.",
    "fm_ld_ml_low": "The FM saw a possible clause that the ML model rated low, so a person should check it.",
    "fm_uncertain": "The FM leaned towards LD but was unsure or could not quote the passage.",
    "ml_fm_disagree": "The ML model and the FM disagree on a passage.",
    "fm_unavailable": "The FM check failed, so a person should look.",
    "ml_uncertain": "The ML model found a borderline passage (FM check not enabled).",
    "no_evidence": "Nothing found. This does not mean the contract has no such clause.",
}

def decide(max_p, fm, T_BYPASS, T_REVIEW, FM_CONF=0.5, T_FLAG_MIN=0.0):
    """max_p: highest ML probability in the contract. fm: FMVerifier.verify() output, or None in ML-only mode."""
    if max_p >= T_BYPASS:
        return "FLAG", "ml_high"
    if fm is None:
        return ("REVIEW", "ml_uncertain") if max_p >= T_REVIEW else ("NO_FLAG", "no_evidence")
    if fm["status"] != "ok":
        return "REVIEW", "fm_unavailable"
    ld = [r for r in fm["results"].values() if r["label"] == "LD"]
    if any(r["valid"] and r["confidence"] >= FM_CONF for r in ld):
        return ("FLAG", "fm_confirmed") if max_p >= T_FLAG_MIN else ("REVIEW", "fm_ld_ml_low")
    if ld:
        return "REVIEW", "fm_uncertain"
    if max_p >= T_REVIEW:
        return "REVIEW", "ml_fm_disagree"
    return "NO_FLAG", "no_evidence"

def rank(n_chunks, probs, fm, FM_CONF=0.5):
    """Order in which passages are shown: FM-confirmed passages first (by FM confidence), then by ML score."""
    conf = {}
    if fm and fm.get("results"):
        conf = {c: r["confidence"] for c, r in fm["results"].items()
                if r["label"] == "LD" and r["valid"] and r["confidence"] >= FM_CONF}
    return sorted(range(n_chunks), key=lambda i: (i not in conf, -conf.get(i, 0.0), -probs[i]))
