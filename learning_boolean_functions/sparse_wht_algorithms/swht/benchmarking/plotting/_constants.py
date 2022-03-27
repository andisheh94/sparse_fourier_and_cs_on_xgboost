"""
Plotting utilities constants
----------------------------

Defines constant values useful throughout the data visualization process.
"""

from typing import OrderedDict
from numpy import uint64, int32, float64
from pathlib import Path


options_types = {'index': uint64, 'run': uint64, 'section': str,
    'time': uint64, 'error': int32, 'samples': uint64, 'n': uint64, 'K': uint64,
    'C': float64, 'ratio': float64, 'robust_iters': uint64, 'cs_bins': uint64,
    'cs_iters': uint64, 'cs_ratio': float64, 'degree': uint64, 'algorithm': str,
    'ops': float64
}

profile_sections = OrderedDict()
profile_sections["init"] = "Initialization"
profile_sections["hasher"] = "M and H"
profile_sections["current_hash"] = "Hash estimate"
profile_sections["robust_init"] = "Ready robust"
profile_sections["time_index"] = "Query index"
profile_sections["query"] = "Signal query"
profile_sections["transform"] = "Fast WHT"
profile_sections["detection"] = "Detect frequency"
profile_sections["robust_retrieval"] = "Robust detect"
profile_sections["updates"] = "Update estimate"

ALGO_MAP = {'naive': "naive", 'randbin': "random binning", 'reedsolo': "reed-solomon"}

HASHING_OPS = None
HASHING_BOUND = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_SOURCES = PROJECT_ROOT.parent / "results_backup"
RESULTS_DIR = PROJECT_ROOT / "benchmarking" / "data"
PLOTS_DIR = PROJECT_ROOT / "benchmarking" / "plots"
