from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / 'assets'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
