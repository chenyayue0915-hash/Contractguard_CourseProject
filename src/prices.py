"""Model prices for choosing the FM and for the business trade-off.

Three layers, best first:
  1. live      — OpenRouter's public model list (https://openrouter.ai/api/v1/models, no key needed),
                 cached in results/openrouter_prices_live.json and refreshed when older than 24 h;
  2. snapshot  — the OpenRouter prices written in config/models.json on `prices_checked_on`;
  3. official  — the model maker's own list price (also in config/models.json, with its source URL).
The cost that actually counts in the evaluation is the one OpenRouter bills per call (fm_verify.py logs it).

    python src/prices.py                        # table from cache/snapshot
    python src/prices.py --refresh              # pull live prices first
    python src/prices.py --refresh --update-snapshot   # also write live prices back into config/models.json
    python src/prices.py --tokens-in 3900 --tokens-out 750 --contracts-per-month 4000
"""
import argparse, datetime as dt, json, time
from pathlib import Path
from config import ROOT, load_models

LIVE_CACHE = ROOT / "results" / "openrouter_prices_live.json"
MAX_AGE_H = 24

def fetch_live(timeout=30):
    """{id: {"in", "out", "tools", "name"}} in USD per 1M tokens, straight from OpenRouter."""
    import requests
    data = requests.get("https://openrouter.ai/api/v1/models", timeout=timeout).json().get("data", [])
    return {m["id"]: {"in": float(m.get("pricing", {}).get("prompt") or 0) * 1e6,
                      "out": float(m.get("pricing", {}).get("completion") or 0) * 1e6,
                      "tools": "tools" in (m.get("supported_parameters") or []),
                      "name": m.get("name", m["id"])} for m in data}

def live_prices(refresh=False, max_age_h=MAX_AGE_H, fetch=None):
    """Live prices with a disk cache. Returns (prices or {}, 'live <UTC time>' | 'none')."""
    cached = json.loads(LIVE_CACHE.read_text()) if LIVE_CACHE.exists() else None
    fresh = cached and time.time() - cached["fetched_at"] < max_age_h * 3600
    if refresh or not fresh:
        try:
            prices = (fetch or fetch_live)()
            cached = {"fetched_at": time.time(), "prices": prices}
            LIVE_CACHE.parent.mkdir(exist_ok=True); LIVE_CACHE.write_text(json.dumps(cached))
        except Exception as e:                                   # offline: fall back to the old cache or the snapshot
            if not cached:
                return {}, f"none (offline: {type(e).__name__})"
    stamp = dt.datetime.fromtimestamp(cached["fetched_at"], dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return cached["prices"], f"live {stamp}"

def price_table(refresh=False, fetch=None):
    cfg = load_models(); live, src = live_prices(refresh, fetch=fetch); rows = []
    for m in cfg["candidates"]:
        lv = live.get(m["id"]); snap = m.get("openrouter") or {}; off = m.get("official") or {}
        orp = (lv["in"], lv["out"]) if lv else (snap.get("in"), snap.get("out"))
        rows.append({"id": m["id"], "label": m.get("label", m["id"]), "tier": m.get("tier", ""), "enabled": m.get("enabled", True),
                     "official_in": off.get("in"), "official_out": off.get("out"),
                     "official_note": off.get("note") or m.get("official_note", ""), "official_source": off.get("source", ""),
                     "openrouter_in": orp[0], "openrouter_out": orp[1],
                     "price_source": src if lv else f"snapshot {cfg.get('prices_checked_on', '?')}",
                     "on_openrouter": (m["id"] in live) if live else None,
                     "tools": lv["tools"] if lv else None})
    return rows, src

def price_for(model, refresh=False, fetch=None):
    """Best available (in, out) USD per 1M tokens for any OpenRouter id, or None."""
    live, _ = live_prices(refresh, fetch=fetch)
    if model in live: return live[model]["in"], live[model]["out"]
    for m in load_models()["candidates"]:
        if m["id"] == model:
            p = m.get("openrouter") or m.get("official")
            return (p["in"], p["out"]) if p else None
    return None

def call_cost(p_in, p_out, tokens_in, tokens_out):
    return (tokens_in * p_in + tokens_out * p_out) / 1e6

def update_snapshot(rows):
    path = ROOT / "config" / "models.json"; cfg = json.loads(path.read_text())
    by = {r["id"]: r for r in rows if r["price_source"].startswith("live")}
    for m in cfg["candidates"]:
        if m["id"] in by:
            m["openrouter"] = {"in": round(by[m["id"]]["openrouter_in"], 4), "out": round(by[m["id"]]["openrouter_out"], 4)}
    cfg["prices_checked_on"] = dt.date.today().isoformat()
    path.write_text(json.dumps(cfg, indent=2) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true"); ap.add_argument("--update-snapshot", action="store_true")
    typ = load_models().get("typical_call", {})
    ap.add_argument("--tokens-in", type=int, default=typ.get("tokens_in", 3800))
    ap.add_argument("--tokens-out", type=int, default=typ.get("tokens_out", 800))
    ap.add_argument("--contracts-per-month", type=int, default=4000)
    ap.add_argument("--from-log", action="store_true", help="use the mean tokens per call measured in results/fm_calls.jsonl")
    ap.add_argument("--calls", type=int, default=89, help="FM calls in one compare_models.py run (dev subset = 89)")
    a = ap.parse_args()
    if a.from_log and (ROOT / "results" / "fm_calls.jsonl").exists():
        live = [json.loads(l) for l in open(ROOT / "results" / "fm_calls.jsonl")]
        live = [r for r in live if not r.get("cached") and r.get("status") == "ok" and r.get("in_tokens")]
        if live:
            a.tokens_in = round(sum(r["in_tokens"] for r in live) / len(live))
            a.tokens_out = round(sum(r["out_tokens"] for r in live) / len(live))
            print(f"Measured from {len(live)} logged calls: {a.tokens_in:,} tokens in, {a.tokens_out:,} out per call "
                  "(thinking models such as GPT-5 / Gemini 3.x / DeepSeek V4 usually produce 2-4x more output).")
    rows, src = price_table(a.refresh)
    if a.update_snapshot:
        if src.startswith("live"): update_snapshot(rows); print("config/models.json snapshot updated")
        else: print("No live prices available; snapshot left unchanged.")
    for r in rows:
        r["cost_per_contract"] = call_cost(r["openrouter_in"], r["openrouter_out"], a.tokens_in, a.tokens_out)
        r["monthly_cost"] = r["cost_per_contract"] * a.contracts_per_month
        r["run_cost"] = r["cost_per_contract"] * a.calls
        r["price_gap"] = ("" if r["official_in"] is None else
                          ("differs from official" if abs(r["openrouter_in"] - r["official_in"]) > 1e-6 or
                           abs(r["openrouter_out"] - r["official_out"]) > 1e-6 else "same as official"))
    rows.sort(key=lambda r: r["cost_per_contract"])
    fmt = lambda v: "–" if v is None else f"${v:.2f}"
    shown = src if src.startswith("live") else f"OpenRouter snapshot {load_models().get('prices_checked_on')} (live refresh failed: {src.replace('none (offline: ', '').rstrip(')')})"
    head = (f"Prices: {shown}. Assumed FM call: {a.tokens_in:,} tokens in + {a.tokens_out:,} out (one per contract, "
            f"bypassed contracts cost $0). Scenario: {a.contracts_per_month:,} contracts per month.\n\n")
    lines = [f"| model | tier | official in / out | OpenRouter in / out | est. cost per contract | per month | one comparison run ({a.calls} calls) | notes |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        note = "; ".join(x for x in [r["price_gap"] if r["price_gap"] == "differs from official" else "", r["official_note"],
                                     "" if r["on_openrouter"] in (None, True) else "NOT listed on OpenRouter",
                                     "" if r["tools"] in (None, True) else "no tool calling"] if x)
        lines.append(f"| {r['label']} | {r['tier']} | {fmt(r['official_in'])} / {fmt(r['official_out'])} | "
                     f"{fmt(r['openrouter_in'])} / {fmt(r['openrouter_out'])} | ${r['cost_per_contract']:.4f} | "
                     f"${r['monthly_cost']:,.2f} | ${r['run_cost']:.2f} | {note} |")
    table = head + "\n".join(lines) + "\n\nPrices are USD per 1M tokens. Sources: " + \
        ", ".join(sorted({r["official_source"] for r in rows if r["official_source"]})) + ", https://openrouter.ai/api/v1/models\n"
    (ROOT / "results" / "price_table.md").write_text(table); print(table)
