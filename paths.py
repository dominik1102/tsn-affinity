# project root = parent of this script's directory (bin/)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "resources" / "datasets"
RUNS = ROOT / "runs"
