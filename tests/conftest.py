"""pytest setup: puts src/ on the import path so the tests can import the modules directly."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
