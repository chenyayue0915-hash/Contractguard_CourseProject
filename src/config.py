"""Local configuration: API keys come from the git-ignored `.env` file (never from code or the repo),
the list of candidate models comes from `config/models.json` (committed, edit freely)."""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_env(path=ROOT / ".env"):
    """Minimal .env reader (KEY=value per line). Values already set in the shell win."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        if v and not os.environ.get(k):                            # an empty shell variable does not block .env
            os.environ[k] = v

def key_problem(name="OPENROUTER_API_KEY", path=ROOT / ".env"):
    """Plain-language reason why a key is missing, or None if it is set."""
    if os.environ.get(name):
        return None
    if not Path(path).exists():
        return f"{path} does not exist. Copy .env.example to .env and paste your key after {name}="
    return (f"{name} in {path} is empty. Paste the key right after the = sign and SAVE the file (Cmd+S). "
            "A key typed into the Streamlit sidebar only works inside the app, not for these scripts.")

def load_models(path=ROOT / "config" / "models.json"):
    return json.loads(Path(path).read_text())

def model_extra(model_id):
    """Per-model request options from config/models.json (e.g. {"reasoning": {"effort": "low"}})."""
    for m in load_models().get("candidates", []):
        if m["id"] == model_id:
            return m.get("extra") or {}
    return {}

load_env()
