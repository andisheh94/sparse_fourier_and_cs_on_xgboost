"""
Data filtering and selection
----------------------------

Provides various data selection and grouping functions.
"""

from pandas import DataFrame, Series, concat
from typing import Dict, Any
from functools import reduce
from operator import and_

from ._constants import HASHING_OPS


def filter_data(data: DataFrame, filter: Dict[str, Any]) -> DataFrame:
    """Selects only rows holding all the values per column given by the filter."""
    return data[reduce(and_, [data[column] == value for column, value in filter.items()])]


def get_comparative_table(data: DataFrame, prefix: str, old_data: DataFrame, old_prefix: str) -> DataFrame:
    """Merges two dataframes after changing their algorithm names"""
    labeled_data = data.copy()
    labeled_data['algorithm'] = prefix + " " + labeled_data['algorithm']
    old_labeled_data = old_data.copy()
    old_labeled_data['algorithm'] = old_prefix + " " + old_labeled_data['algorithm']
    return concat((labeled_data, old_labeled_data), ignore_index=True, copy=False)


def select_performance(data: DataFrame, selection: Dict[str, Any]) -> DataFrame:
    """
    Isolates hashing section timing and compute its average performance for a
    fixed number of operations.
    """
    selection['section'] = 'time_index'
    data = filter_data(data, selection)
    avg_times = data['time'] / data['samples']
    data['performance'] = HASHING_OPS / avg_times
    return data
