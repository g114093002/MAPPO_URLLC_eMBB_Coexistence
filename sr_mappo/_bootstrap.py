from pathlib import Path
import sys

PACKAGE_DIR = Path(__file__).resolve().parent
SEARCH_ROOTS = [PACKAGE_DIR.parent / 'Greedy', PACKAGE_DIR.parent]
GREEDY_ROOT = None
for candidate in SEARCH_ROOTS:
    if (candidate / 'config.py').exists() and (candidate / 'simulation.py').exists():
        GREEDY_ROOT = candidate
        break
if GREEDY_ROOT is None:
    raise RuntimeError('Could not locate Greedy project root for sr_mappo bootstrap.')
root_str = str(GREEDY_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
