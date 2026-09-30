"""FM verification: ONE batched call to a rented foundation model per contract.

Providers: "openrouter" (default; any model OpenRouter serves, key OPENROUTER_API_KEY in .env) or
"anthropic" (Claude directly, key ANTHROPIC_API_KEY). Cost per call is taken from the provider's own
accounting when available (OpenRouter `usage.cost`), otherwise from token counts x listed price.

The model receives the ~20 candidate chunks picked by the ML retriever and returns, through a forced
tool call (structured output), one JSON record per chunk: {chunk_id, label, confidence, evidence}.

Guardrails implemented here (not just described):
  * contract text is treated as untrusted data: tag characters are neutralised and the system prompt
    tells the model never to follow instructions found inside passages (OWASP LLM01);
  * the output schema is enforced by the tool definition AND re-validated in code (OWASP LLM05);
  * an LD label only counts if its evidence is an exact quote from that chunk; otherwise it is "invalid";
  * no free text from the model is ever shown to the user;
  * every call is logged (tokens, cost, latency) and can be cached so results reproduce without an API key.
"""
import hashlib, json, os, re, threading, time
from collections import Counter
from pathlib import Path
import config                                                    # loads .env

DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Anthropic-direct list prices, USD per million tokens (input, output) — platform.claude.com/docs/en/about-claude/pricing
PRICES = {"claude-haiku-4-5-20251001": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0)}
KEY_ENV = {"openrouter": "OPENROUTER_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
MAX_PASSAGES = 30
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

TOOL = {
    "name": "report_labels",
    "description": "Report exactly one label for every passage you were given.",
    "input_schema": {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer"},
                "label": {"type": "string", "enum": ["LD", "NOT_LD"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string", "description": "exact quote from the passage, or empty for NOT_LD"}},
            "required": ["chunk_id", "label", "confidence", "evidence"]}}},
        "required": ["results"]},
}

def load_system_prompt(prompt="zero"):
    base = (PROMPT_DIR / "system_zero_shot.txt").read_text()
    if prompt == "few":
        ex = json.loads((PROMPT_DIR / "few_shot_examples.json").read_text())
        lines = ["", "Examples (from the training split):"]
        for e in ex:
            lines.append(f'Passage: "{e["text"]}"\n-> label {e["label"]}' + (f', evidence "{e["evidence"]}"' if e["evidence"] else ""))
        base += "\n".join(lines)
    elif prompt != "zero":
        raise ValueError("prompt must be 'zero' or 'few'")
    return base

def sanitize(text):
    """Neutralise anything that could open or close our passage tags."""
    return re.sub(r"\s+", " ", text.replace("<", "(").replace(">", ")")).strip()

def build_user_message(passages):
    body = "\n".join(f'<passage id="{cid}">{sanitize(t)}</passage>' for cid, t in passages)
    ids = ", ".join(str(cid) for cid, _ in passages)
    return (f"<passages>\n{body}\n</passages>\nThere are {len(passages)} passages (ids: {ids}). "
            f"Call the report_labels tool once with exactly one record for EACH of these ids; never return an empty list.")

def _norm(s):
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip().lower()

def evidence_ok(evidence, passage_text):
    ev = _norm(evidence)
    return len(ev) >= 8 and ev in _norm(sanitize(passage_text))

def validate(tool_input, passages):
    """Re-check the model output in code. Returns (status, {chunk_id: record})."""
    texts = dict(passages)
    if not isinstance(tool_input, dict) or not isinstance(tool_input.get("results"), list):
        return "schema_error", {}
    out = {}
    for r in tool_input["results"]:
        try:
            cid = int(r["chunk_id"]); label = r["label"]; conf = float(r["confidence"]); ev = str(r.get("evidence", ""))
        except (KeyError, TypeError, ValueError):
            continue
        if cid not in texts or label not in ("LD", "NOT_LD") or not 0 <= conf <= 1 or cid in out:
            continue
        valid = label == "NOT_LD" or evidence_ok(ev, texts[cid])
        out[cid] = {"label": label, "confidence": conf, "evidence": ev if label == "LD" else "", "valid": valid}
    missing = [c for c in texts if c not in out]
    return ("ok" if not missing else ("schema_error" if len(missing) == len(texts) else "partial")), out

class AnthropicClient:
    """Claude called directly through Anthropic's SDK."""
    def __init__(self, api_key):
        import anthropic                                        # imported lazily: only needed for live calls
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(self, model, system, user, max_tokens, temperature):
        r = self.client.messages.create(model=model, max_tokens=max_tokens, system=system, temperature=temperature,
                                        tools=[TOOL], tool_choice={"type": "tool", "name": TOOL["name"]},
                                        messages=[{"role": "user", "content": user}])
        block = next((b for b in r.content if getattr(b, "type", "") == "tool_use"), None)
        i, o = r.usage.input_tokens, r.usage.output_tokens
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return (block.input if block else None), i, o, (i * pin + o * pout) / 1e6

def _records(obj):
    """Pull label records out of whatever shape a model used: {"results": [...]}, a list, or one bare record."""
    if isinstance(obj, str):
        try: obj = json.loads(obj)
        except json.JSONDecodeError: return []
    if isinstance(obj, dict) and "results" in obj: return _records(obj["results"])
    if isinstance(obj, dict) and "chunk_id" in obj: return [obj]
    if isinstance(obj, list): return [r for x in obj for r in _records(x)]
    return []

def parse_openrouter_response(data):
    """OpenAI-format response -> (tool_input, in_tokens, out_tokens, cost_usd or None).
    Merges records from EVERY tool call (some models split the answer over several calls or return one call per
    passage) and falls back to JSON in the message text for models that answer in plain text."""
    usage = data.get("usage") or {}
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    recs, saw_call = [], False
    for tc in msg.get("tool_calls") or []:
        saw_call = True
        recs += _records((tc.get("function") or {}).get("arguments", ""))
    if not recs and isinstance(msg.get("content"), str):
        txt = re.sub(r"^```(?:json)?|```$", "", msg["content"].strip(), flags=re.M).strip()
        m = re.search(r"[\[{].*[\]}]", txt, flags=re.S)
        recs = _records(m.group(0)) if m else []
    ti = {"results": recs} if (recs or saw_call) else None
    cost = usage.get("cost")
    return ti, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), \
        (float(cost) if cost is not None else None)

class OpenRouterClient:
    """Any model on OpenRouter through its OpenAI-compatible endpoint (plain HTTPS, no SDK needed)."""
    def __init__(self, api_key, extra=None, timeout=120):
        self.api_key, self.extra, self.timeout = api_key, extra or {}, timeout

    def call(self, model, system, user, max_tokens, temperature):
        import requests
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "tools": [{"type": "function", "function": {"name": TOOL["name"], "description": TOOL["description"],
                                                            "parameters": TOOL["input_schema"]}}],
                "tool_choice": {"type": "function", "function": {"name": TOOL["name"]}},
                "provider": {"require_parameters": True},        # only route to providers that support tool calling
                "usage": {"include": True}}                        # ask OpenRouter to return the billed cost
        body.update(self.extra)                                    # per-model options from config/models.json
        r = requests.post(f"{OPENROUTER_URL}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}", "X-Title": "ContractGuard (NTU PE6201)"})
        r.raise_for_status()
        data = r.json()
        ti, i, o, cost = parse_openrouter_response(data)
        if not ti or not ti.get("results"):                         # keep the raw answer so the failure can be diagnosed
            try:
                with open(Path(__file__).resolve().parent.parent / "results" / "fm_debug.jsonl", "a") as f:
                    f.write(json.dumps({"ts": time.time(), "model": model, "raw": data})[:20000] + "\n")
            except OSError:
                pass
        if cost is None:                                           # fall back to live / snapshot list price
            from prices import price_for, call_cost
            p = price_for(model)
            cost = call_cost(p[0], p[1], i, o) if p else 0.0
        return ti, i, o, cost

class FMVerifier:
    def __init__(self, model=DEFAULT_MODEL, prompt="zero", samples=1, client=None,
                 cache_path=None, log_path=None, api_key=None, provider=DEFAULT_PROVIDER, extra=None):
        self.model, self.prompt, self.samples, self.provider = model, prompt, samples, provider
        self.extra = extra if extra is not None else (config.model_extra(model) if provider == "openrouter" else {})
        self.system = load_system_prompt(prompt)
        self.cache_path, self.log_path = cache_path, log_path
        self.cache = {}
        if cache_path and Path(cache_path).exists():
            for line in open(cache_path):
                rec = json.loads(line); self.cache[rec["key"]] = rec
        self._client = client
        self._api_key = api_key or os.environ.get(KEY_ENV.get(provider, ""), "")
        self._lock = threading.Lock()                            # compare_models.py calls verify() from threads
        self._session = {}                                        # every answer this run (empty ones too): never pay twice

    @property
    def live(self):
        return self._client is not None or bool(self._api_key)

    def _client_or_none(self):
        if self._client is None and self._api_key:
            self._client = (OpenRouterClient(self._api_key, self.extra) if self.provider == "openrouter"
                            else AnthropicClient(self._api_key))
        return self._client

    def _key(self, user, sample):
        raw = json.dumps(["v2", self.provider, self.model, self.extra, self.prompt, self.system, user, self.samples, sample])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _one(self, user, sample, n):
        """One API call (or cache hit). Returns (tool_input, in_tok, out_tok, cost_usd, cached, error)."""
        key = self._key(user, sample)
        if key in self.cache:
            c = self.cache[key]; return c["tool_input"], c["in"], c["out"], c.get("cost", 0.0), True, None
        if key in self._session:
            return (*self._session[key], True, None)
        client = self._client_or_none()
        if client is None:
            return None, 0, 0, 0.0, False, "no_api_key_and_not_cached"
        try:
            r = client.call(self.model, self.system, user, max_tokens=150 * n + 1000,   # room for hidden reasoning tokens
                            temperature=0.0 if self.samples == 1 else 1.0)
        except Exception as e:                                   # network, auth, rate limit, unknown model ...
            return None, 0, 0, 0.0, False, f"api_error: {type(e).__name__}: {str(e)[:120]}"
        ti, i, o = r[:3]
        cost = r[3] if len(r) > 3 and r[3] is not None else (i * PRICES.get(self.model, (0, 0))[0] +
                                                              o * PRICES.get(self.model, (0, 0))[1]) / 1e6
        self._session[key] = (ti, i, o, cost)
        if self.cache_path and ti and ti.get("results"):          # empty answers are not cached, so a rerun retries them
            with self._lock:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "tool_input": ti, "in": i, "out": o, "cost": cost}) + "\n")
                self.cache[key] = {"tool_input": ti, "in": i, "out": o, "cost": cost}
        return ti, i, o, cost, False, None

    def verify(self, passages, tag=""):
        """passages: list of (chunk_id, text). Returns a dict with status, per-chunk results, tokens, cost, latency."""
        passages = list(passages)[:MAX_PASSAGES]
        user = build_user_message(passages)
        t0 = time.time(); runs = []; tin = tout = 0; cost = 0.0; cached_all = True; err = None
        for s in range(self.samples):
            ti, i, o, c, cached, e = self._one(user, s, len(passages))
            tin += i; tout += o; cost += c; cached_all &= cached
            if e: err = e; break
            runs.append(validate(ti, passages))
        repaired = False
        if not err and len(runs) == 1 and runs[0][0] != "ok":
            # Repair: weaker models sometimes return an empty or incomplete list. Ask again for the missing ids only,
            # first all together, then in batches of 5 (small batches are answered far more reliably).
            merged = dict(runs[0][1])
            missing = [(cid, t) for cid, t in passages if cid not in merged]
            batches = [missing] + [missing[k:k + 5] for k in range(0, len(missing), 5)] if len(missing) > 5 else [missing]
            for n_try, batch in enumerate(batches):
                batch = [(cid, t) for cid, t in batch if cid not in merged]
                if not batch: continue
                ti2, i2, o2, c2, cached2, e2 = self._one(build_user_message(batch), 100 + n_try, len(batch))
                tin += i2; tout += o2; cost += c2; cached_all &= cached2
                if not e2:
                    merged.update(validate(ti2, batch)[1])
            runs = [("ok" if len(merged) == len(passages) else "partial", merged)]; repaired = True
        latency = time.time() - t0
        if err or not runs:
            status, results = (err or "api_error"), {}
        elif len(runs) == 1:
            status, results = runs[0]
        else:                                                    # self-consistency: majority vote per chunk
            results = {}
            for cid, _ in passages:
                votes = [r[1][cid] for r in runs if cid in r[1]]
                if not votes: continue
                lab = Counter(v["label"] for v in votes).most_common(1)[0][0]
                same = [v for v in votes if v["label"] == lab]
                ev = next((v for v in same if v["valid"]), same[0])
                results[cid] = {"label": lab, "confidence": sum(v["confidence"] for v in same) / len(votes),
                                "evidence": ev["evidence"], "valid": ev["valid"]}
            status = "ok" if len(results) == len(passages) else "partial"
        out = {"status": status, "results": results, "in_tokens": tin, "out_tokens": tout,
               "cost_usd": cost, "latency_s": latency, "cached": cached_all and not err, "repaired": repaired}
        if self.log_path:
            with self._lock, open(self.log_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "tag": tag, "provider": self.provider, "model": self.model,
                                    "prompt": self.prompt,
                                    "samples": self.samples, "n_passages": len(passages), "status": status,
                                    "in_tokens": tin, "out_tokens": tout, "cost_usd": round(cost, 6),
                                    "latency_s": round(latency, 3), "cached": out["cached"],
                                    "repaired": repaired}) + "\n")
        return out
