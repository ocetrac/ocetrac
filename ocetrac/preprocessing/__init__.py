# ============================================================
# ocetrac/preprocessing/__init__.py
# ============================================================
"""
ocetrac.preprocessing — Preprocessing pipeline
 
Re-exports the public API so callers can write:
 
    from ocetrac.preprocessing import clean_binary
    from ocetrac.preprocessing import compute_anomalies, threshold_features
    from ocetrac.preprocessing import cesm2_lens_utils
"""
from .preprocessing import (
    clean_binary,
    compute_anomalies,
    threshold_features,
)
from . import cesm2_lens_utils
 
__all__ = [
    "clean_binary",
    "compute_anomalies",
    "threshold_features",
    "cesm2_lens_utils",
]
 