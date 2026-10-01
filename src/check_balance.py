"""How much OpenRouter credit is left, and how much this project has spent.

    python src/check_balance.py

Reads OPENROUTER_API_KEY from .env. Uses OpenRouter's key endpoint (spend and limit of THIS key) and, if the key is
allowed to, the account credits endpoint. Also totals the billed cost logged in results/fm_calls.jsonl.
"""
import json, os
from pathlib import Path
import config                                                    # loads .env

URL = "https://openrouter.ai/api/v1"

def get(path, key):
    import requests
    r = requests.get(URL + path, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})

def money(v):
    return "not set (no limit on this key)" if v is None else f"${float(v):,.4f}"

if __name__ == "__main__":
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit(config.key_problem() or "OPENROUTER_API_KEY is missing.")
    code, body = get("/key", key)
    d = body.get("data") or {}
    if code == 200 and d:
        print("This API key (from OpenRouter):")
        print(f"  label            : {d.get('label', '')}")
        print(f"  spent so far     : {money(d.get('usage'))}")
        print(f"  spending limit   : {money(d.get('limit'))}")
        if d.get("limit") is not None:
            print(f"  left on this key : {money(d.get('limit_remaining'))}")
    else:
        print(f"Key endpoint answered HTTP {code}: {json.dumps(body)[:200]}")
    code, body = get("/credits", key)
    c = body.get("data") or {}
    if code == 200 and c:
        total, used = float(c.get("total_credits") or 0), float(c.get("total_usage") or 0)
        print("Whole OpenRouter account:")
        print(f"  credits bought   : ${total:,.4f}\n  used             : ${used:,.4f}\n  left             : ${total - used:,.4f}")
    else:
        print(f"(Account credits not visible with this key: HTTP {code}. The key numbers above are what matters.)")
    log = config.ROOT / "results" / "fm_calls.jsonl"
    if log.exists():
        rows = [json.loads(l) for l in open(log)]
        live = [r for r in rows if not r.get("cached")]
        print(f"This project's own log: {len(live)} paid calls, ${sum(r.get('cost_usd', 0) for r in live):,.4f} billed")
