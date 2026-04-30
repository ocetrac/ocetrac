# ocetrac/tests/conftest.py
#
# Adds the repo root to sys.path so that `import ocetrac.DeepTrack`
# resolves to ocetrac/ocetrac/DeepTrack/.
#
# Directory layout assumed:
#
#   ocetrac/               ← repo root
#   ├── tests/
#   │   ├── conftest.py    ← this file
#   │   └── test_deeptrack.py
#   └── ocetrac/           ← Python package
#       ├── __init__.py
#       ├── DeepTrack/
#       ├── preprocessing/
#       └── SurfTrack/
 
import sys
import os
 
# tests/ → repo root (one level up)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)