"""ContractGuard — Streamlit app.   Run:  streamlit run app.py
Flags liquidated-damages / termination-fee clauses in a vendor contract. Flag only: no rewriting, no legal advice."""
import html, os, sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from extract import extract_text, check_text          # noqa: E402
from pipeline import ContractGuard                    # noqa: E402
from router import REASONS                            # noqa: E402
from config import load_models, load_env              # noqa: E402

st.set_page_config(page_title="ContractGuard", page_icon="📄", layout="centered")

@st.cache_resource
def load_guard():
    return ContractGuard()

def merge_passages(top):
    """Neighbouring windows overlap by 50%; merge overlapping ones so the same sentence is not shown twice."""
    merged = []
    for t in top:
        for m in merged:
            if t["start"] < m["end"] and m["start"] < t["end"]:
                m["start"], m["end"] = min(m["start"], t["start"]), max(m["end"], t["end"])
                m["p"] = max(m["p"], t["p"]); m["parts"].append(t); break
        else:
            merged.append({"start": t["start"], "end": t["end"], "p": t["p"], "parts": [t]})
    return merged

def highlight(text, quotes):
    safe = html.escape(text)
    for q in quotes:
        if q:
            qs = html.escape(q)
            i = safe.lower().find(qs.lower())
            if i >= 0:
                safe = safe[:i] + "<mark>" + safe[i:i + len(qs)] + "</mark>" + safe[i + len(qs):]
    return safe

guard = load_guard()
S = guard.settings

st.title("ContractGuard")
st.caption("Checks a vendor contract for liquidated-damages and termination-fee clauses before you sign. "
           "It only points at passages to read. It does not give legal advice.")

@st.cache_data(ttl=3600, show_spinner=False)
def prices():
    from prices import price_table
    rows, src = price_table()                       # live OpenRouter prices (cached 24 h), else the snapshot
    return {r["id"]: r for r in rows}, src

with st.sidebar:
    st.header("FM settings")
    load_env()                                                              # re-read .env on every rerun (no restart needed)
    cfg = load_models()
    env_key = os.environ.get("OPENROUTER_API_KEY", "")                      # read from the git-ignored .env file
    if env_key:
        st.success("OpenRouter key loaded from .env")
    typed = st.text_input("OpenRouter API key", value="", type="password",
                          help="Best: put it in .env (see .env.example). A key typed here lives only in this browser session.")
    key = typed or env_key
    labels = {m["id"]: m.get("label", m["id"]) for m in cfg["candidates"]}
    tiers = {m["id"]: m.get("tier", "") for m in cfg["candidates"]}
    options = list(labels) + ["Other (type an OpenRouter model id)"]
    frozen_model = S.get("MODEL", cfg["default_model"])                      # the model the thresholds were tuned for
    default = options.index(frozen_model) if frozen_model in options else 0
    choice = st.selectbox("Model", options, index=default,                   # always selectable, so prices can be compared
                          format_func=lambda m: (f"{labels[m]} · {tiers[m]}" + (" · frozen" if m == frozen_model else ""))
                          if m in labels else m)
    model = st.text_input("OpenRouter model id", value="") if choice.startswith("Other") else choice
    if model and model != frozen_model and "MODEL" in S:
        st.warning(f"The REVIEW / FLAG thresholds were tuned for {labels.get(frozen_model, frozen_model)}. "
                   "Another model works, but its labels are not the evaluated configuration.")
    use_fm = st.toggle("FM verification", value=bool(key), disabled=not key,
                       help="Sends about 20 candidate passages to the chosen model in one call.")
    if not key:
        st.info("No OpenRouter key yet, so FM verification is off and the app runs the ML model only. "
                "Paste the key after OPENROUTER_API_KEY= in the .env file and save it, or type it above.")
    ptab, psrc = prices()
    p = ptab.get(model)
    if p:
        typ = cfg.get("typical_call", {"tokens_in": 3800, "tokens_out": 800})
        est = (typ["tokens_in"] * p["openrouter_in"] + typ["tokens_out"] * p["openrouter_out"]) / 1e6
        off = ("–" if p["official_in"] is None else f"${p['official_in']:.2f} / ${p['official_out']:.2f}")
        st.caption(f"**{p['tier']}** · OpenRouter ${p['openrouter_in']:.2f} in / ${p['openrouter_out']:.2f} out per 1M tokens "
                   f"({p['price_source']}) · official {off} · ≈ ${est:.4f} per contract"
                   + (f" · {p['official_note']}" if p["official_note"] else ""))
    with st.expander("Price table (all candidates)"):
        typ = cfg.get("typical_call", {"tokens_in": 3800, "tokens_out": 800})
        rows = sorted(ptab.values(), key=lambda r: r["openrouter_in"] * typ["tokens_in"] + r["openrouter_out"] * typ["tokens_out"])
        st.markdown("| model | $/1M in | $/1M out | ≈ $/contract |\n|---|---|---|---|\n" + "\n".join(
            f"| {r['label']} | {r['openrouter_in']:.2f} | {r['openrouter_out']:.2f} | "
            f"{(typ['tokens_in'] * r['openrouter_in'] + typ['tokens_out'] * r['openrouter_out']) / 1e6:.4f} |" for r in rows))
        st.caption(f"Prices: {psrc if psrc.startswith('live') else 'snapshot ' + cfg.get('prices_checked_on', '')}. "
                   "Refresh with: python src/prices.py --refresh")
    st.caption(f"Frozen on the development split: FLAG without FM at ML score ≥ {S['T_BYPASS']}, "
               f"{S['N_CAND']} candidates + keyword hits, prompt: {S.get('PROMPT', 'few')}, "
               f"FLAG needs FM confidence ≥ {S.get('FM_CONF')} and ML score ≥ {S.get('T_FLAG_MIN')}.")
    if "T_REVIEW_frozen" not in S:
        st.caption("Hybrid REVIEW threshold not tuned yet (run evaluate_hybrid.py --split dev --freeze).")

tab_up, tab_paste, tab_demo = st.tabs(["Upload", "Paste text", "Demo contracts"])
with tab_up:
    up = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
with tab_paste:
    pasted = st.text_area("Contract text", height=200)
with tab_demo:
    demos = sorted(p.name for p in (ROOT / "demo").glob("*.txt"))
    demo = st.selectbox("Choose a demo contract", ["(none)"] + demos)

if st.button("Check contract", type="primary"):
    if up is not None:
        text, source = extract_text(up.name, up.getvalue()), up.name
    elif pasted.strip():
        text, source = pasted, "pasted text"
    elif demo != "(none)":
        text, source = (ROOT / "demo" / demo).read_text(), demo
    else:
        st.warning("Upload a file, paste text, or choose a demo contract."); st.stop()

    ok, msg = check_text(text)
    if not ok:
        st.error(msg); st.stop()

    fm = None
    if use_fm and key and model:
        from fm_verify import FMVerifier
        fm = FMVerifier(model=model, prompt=S.get("PROMPT", "few"), samples=S.get("SAMPLES", 1), provider="openrouter",
                        api_key=key, cache_path=None, log_path=None)   # uploads are never written to disk
    with st.spinner("Reading the contract…"):
        out = guard.analyze_text(text, fm_verifier=fm)

    label = out["label"]
    if label == "FLAG":
        st.error("**FLAG** — this contract appears to contain a liquidated-damages or termination-fee clause. "
                 "Pause before signing and ask a lawyer about the passages below.")
    elif label == "REVIEW":
        st.warning("**REVIEW** — there may be such a clause, but the system is not sure. "
                   "Read the passages below, and ask a lawyer if any of them makes you pay a fixed sum.")
    else:
        st.info("**NO_FLAG** — nothing was found. This does not mean the contract is free of such clauses; "
                "the passages below are the closest matches.")
    st.caption(f"Why: {REASONS[out['reason']]}  ·  Source: {source}  ·  {msg}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Passages scanned", out["n_chunks"])
    c2.metric("Checked by FM", len(out["candidates"]) if out["fm_used"] else 0)
    c3.metric("FM cost (this contract)", f"${out['cost_usd']:.4f}")
    if out["fm_used"]:
        st.caption(f"Model: {model} · latency {out['latency_s']:.1f} s · at this cost, 1,000 contracts ≈ ${out['cost_usd'] * 1000:,.2f}")
    if fm is not None and not out["fm_used"]:
        st.caption(f"FM not called: an ML score of {out['max_p']:.2f} is above the {S['T_BYPASS']} threshold.")
    if out["fm_used"] and out["fm_status"] != "ok":
        st.caption(f"FM check status: {out['fm_status']} — treated as REVIEW.")

    st.subheader("Passages to read")
    for n, m in enumerate(merge_passages(out["top"]), 1):
        parts = m["parts"]
        fm_lab = next((p for p in parts if p.get("fm_label") == "LD" and p.get("evidence_ok")), None)
        tag = f"ML score {m['p']:.2f}" + (f" · FM: LD ({fm_lab['fm_conf']:.2f})" if fm_lab else "")
        st.markdown(f"**{n}.** <span style='color:gray'>{tag}</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='border-left:3px solid #ccc;padding-left:10px'>"
                    f"{highlight(text[m['start']:m['end']], [p.get('evidence') for p in parts if p.get('evidence_ok')])}</div>",
                    unsafe_allow_html=True)

st.divider()
st.caption("ContractGuard screens for one clause type in English commercial vendor contracts. It is not a lawyer and not "
           "legal advice, and it does not check whether a clause is enforceable. Your contract is processed in memory and "
           "not saved; with FM verification on, about 20 candidate passages are sent to the chosen model via OpenRouter.")
