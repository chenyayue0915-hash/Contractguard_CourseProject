"""Command-line version of the app: check one contract file and print the label, the reason and the passages to read.

    python src/check_contract.py demo/fresh_produce_supply_agreement.txt
    python src/check_contract.py demo/cuad_test_sponsorship_agreement_noLD.txt --no-fm     # ML only, no key needed
    python src/check_contract.py my_contract.pdf --top 5

It runs the same pipeline, frozen settings and frozen model as the Streamlit app and the evaluation (src/pipeline.py),
so what it prints is what the app shows. Like the app, the FM call is live and nothing is cached or logged, unless
--cached is given: then answers are reused from results/fm_cache.jsonl, so a repeated demo run returns at once.
"""
import argparse, os, sys, textwrap
from pathlib import Path
import config                                   # noqa: F401  (reads the git-ignored .env on import)
from extract import extract_text, check_text
from pipeline import ContractGuard, ROOT
from router import REASONS
from common import K_BROAD

COLOUR = {"FLAG": "\033[1;31m", "REVIEW": "\033[1;33m", "NO_FLAG": "\033[1;34m"}
MARK, END = "\033[1;7m", "\033[0m"
LABEL_TEXT = {"FLAG": "this contract appears to contain a liquidated-damages or termination-fee clause.",
              "REVIEW": "there may be such a clause, but the system is not sure. Read the passages below.",
              "NO_FLAG": "nothing was found. This does not mean the contract is free of such clauses."}

def snippet(text, evidence, width=320):
    """The part of a passage worth reading: around the FM's quoted evidence if there is one, else its start."""
    flat = " ".join(text.split())
    if evidence:
        i = flat.lower().find(" ".join(evidence.split()).lower())
        if i >= 0:
            j = i + len(" ".join(evidence.split()))
            a = max(0, i - 120)
            return ("…" if a else "") + flat[a:i] + MARK + flat[i:j] + END + flat[j:j + 80] + "…"
    m = K_BROAD.search(flat)                    # no quote: centre on the clause words the keyword list knows
    if m:
        a = max(0, m.start() - 120)
        return ("…" if a else "") + flat[a:m.start()] + MARK + flat[m.start():m.end()] + END + flat[m.end():m.end() + 160] + "…"
    return flat[:width] + ("…" if len(flat) > width else "")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check one contract for liquidated-damages / termination-fee clauses.")
    ap.add_argument("path", help="a .txt, .pdf or .docx contract")
    ap.add_argument("--no-fm", action="store_true", help="ML model only (no API key needed)")
    ap.add_argument("--cached", action="store_true", help="reuse FM answers from results/fm_cache.jsonl")
    ap.add_argument("--top", type=int, default=3, help="passages to print (the app shows 5)")
    a = ap.parse_args()

    path = Path(a.path)
    text = extract_text(path.name, path.read_bytes())
    ok, msg = check_text(text)
    if not ok:
        sys.exit(msg)
    guard = ContractGuard(); s = guard.settings
    fm = None
    if not a.no_fm:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("No OPENROUTER_API_KEY in .env, so this run uses the ML model only.\n")
        else:
            from fm_verify import FMVerifier
            fm = FMVerifier(model=s.get("MODEL"), prompt=s.get("PROMPT", "zero"), samples=s.get("SAMPLES", 1),
                            provider=s.get("PROVIDER", "openrouter"),
                            cache_path=ROOT / "results" / "fm_cache.jsonl" if a.cached else None, log_path=None)
    out = guard.analyze_text(text, fm_verifier=fm)

    print(f"ContractGuard · {path}")
    print(f"  passages scanned : {out['n_chunks']}")
    print(f"  highest ML score : {out['max_p']:.2f}")
    if out["fm_used"]:
        print(f"  FM check         : {s.get('MODEL')} · {len(out['candidates'])} passages · "
              f"${out['cost_usd']:.4f} · {out['latency_s']:.1f} s" + (" (cached)" if a.cached else ""))
    elif fm is not None:
        print(f"  FM check         : not needed (ML score above {s['T_BYPASS']}) · $0.0000")
    lab = out["label"]
    print(f"\n{COLOUR[lab]}{lab}{END} — {LABEL_TEXT[lab]}")
    print(f"  why: {REASONS[out['reason']]}\n")
    print("Passages to read")
    shown = []
    for t in out["top"]:
        if any(t["start"] < e and b < t["end"] for b, e in shown):      # overlapping window already shown
            continue
        shown.append((t["start"], t["end"]))
        tag = f"ML {t['p']:.2f}" + (f" · FM: LD ({t['fm_conf']:.2f}), quoted" if t.get("fm_label") == "LD" and t.get("evidence_ok") else "")
        body = snippet(text[t["start"]:t["end"]], t.get("evidence") if t.get("evidence_ok") else "")
        print(f" {len(shown)}. {tag}")
        print(textwrap.indent(textwrap.fill(body, 100), "    "))
        if len(shown) == a.top:
            break
    print("\nNot legal advice. ContractGuard flags one clause type; it does not judge enforceability.")
