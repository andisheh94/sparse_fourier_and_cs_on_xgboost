"""
Plotting utilities
------------------

Provide functions to filter, merge and orgnanize data files and to then produce
readable plots depending on given parameters.
"""

from typing import List
from ._scripts import get_speedup_tables, load_and_parse, fetch_times, \
    get_time_plots, get_profiling
from ._constants import ALGO_MAP

CS_ALGOS: List[str] = ALGO_MAP.values()

__all__ = [
    "get_speedup_tables",
    "load_and_parse",
    "fetch_times",
    "get_time_plots",
    "get_profiling",
    "CS_ALGOS"
]
