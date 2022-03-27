"""
Main scripts
------------

Defines function to process the data from file to plot.
"""

from sys import stderr
from typing import Any, Dict, List, OrderedDict, Tuple
from pandas import DataFrame, options as pandas_options
pandas_options.mode.chained_assignment = None

from ._constants import PLOTS_DIR, profile_sections
from ._filtering import filter_data, get_comparative_table
from ._datafiles import load_sources, filter_results, merge_results, results_path, read_results_file
from ._visualizing import line_plot, bar_plot


def __nice_lang_name(title: str):
    """Nice language interface name selector."""
    return title.replace('py', "Python").replace('raw', "C++")

def __nice_pdf_name(title: str):
    """Simple nice pdf filename."""
    return title.replace('=', '').replace(' ', '_') + ".pdf"

def __get_order(branch: str, old_branch: str) -> List[str]:
    """Quick and dirty ordering for the plots."""
    algos = ["naive", "random binning", "reed-solomon"]
    return [f"{old_branch} {algo}" for algo in algos] + [f"{branch} {algo}" for algo in algos]


def fetch_times(branch: str, sectioned: bool = False) -> Dict[Tuple[str, str], DataFrame]:
    """Get dataframes from a branch (sectioned or global)."""
    dir_path = results_path(branch, sectioned=sectioned)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"No {dir_path} directory.")
    frames = {}
    for csv_file in dir_path.glob("*.csv"):
        interface, robustness = csv_file.stem.split('_')
        frames[(interface, robustness)] = read_results_file(csv_file)
    return frames


def load_and_parse(branch: str, purge: bool = True):
    """Load filter and merge the results of a branch."""
    directory = load_sources(branch, purge)
    time_dir = directory / "results"
    if time_dir.is_dir():
        good_results, timeouts, failures = filter_results(time_dir)
        merge_results(time_dir, good_results)
        merge_results(time_dir / "timeouts", timeouts)
        merge_results(time_dir / "failures", failures)
    profiles_dir = directory / "profiles"
    if profiles_dir.is_dir():
        good_results, timeouts, failures = filter_results(profiles_dir)
        merge_results(profiles_dir, good_results)
        merge_results(profiles_dir / "timeouts", timeouts)
        merge_results(profiles_dir / "failures", failures)


def __check_coherence(**parameters) -> Tuple[str, Dict[str, Any]]:
    """Make sure that the filter given is legal."""
    varying = None
    for parameter, value in parameters.items():
        if value is None:
            if varying is not None:
                raise ValueError(f"Only one parameter can vary ('{varying}' and '{parameter}').")
            varying = parameter
    if varying is not None:
        parameters.pop(varying)
    return varying, parameters


def get_time_plots(data: DataFrame, branch: str, old_data: DataFrame, old_branch: str,
        dest: str, title: str, n: int = None, k: int = None, degree: int = None):
    """Create and save a comparative global runtime linear plot."""
    variable_column, selection = __check_coherence(n=n, K=k, degree=degree)
    joined_frame = get_comparative_table(data, branch, old_data, old_branch)
    joined_frame = filter_data(joined_frame, selection)
    file_title = __nice_pdf_name(title)
    destination = PLOTS_DIR / branch / dest
    destination.mkdir(parents=True, exist_ok=True)
    line_plot(destination / file_title, joined_frame, "time", __nice_lang_name(title), variable_column, "[cycles]", __get_order(branch, old_branch))


def get_performance_plots():
    pass # TODO


def get_speedup_tables(data: DataFrame, branch: str, old_data: DataFrame, measure: str,
n: int = None, k: int = None, degree: int = None, dest: str = None):
    """Joins a column from two dataframes and compute their speedup as a table."""
    variable_column, selection = __check_coherence(n=n, K=k, degree=degree)
    data = filter_data(data, selection)[['algorithm', variable_column, measure]]\
        .groupby(['algorithm', variable_column], as_index=True).median()
    old_data = filter_data(old_data, selection)[['algorithm', variable_column, measure]]\
        .groupby(['algorithm', variable_column], as_index=True).median()
    merged_data = data.merge(old_data, how='outer', left_index=True,
        right_index=True, suffixes=['', '_old'])
    speedups = (merged_data[f"{measure}_old"] / merged_data[measure]).unstack(level=1)
    if dest is not None:
        path = PLOTS_DIR / branch / "speedups"
        path.mkdir(parents=True, exist_ok=True)
        speedups.to_csv(path / f"{dest}.csv")
    return speedups


def get_profiling(data: DataFrame, branch: str, title: str, algo: str, n: int, k: int, degree: int):
    """Create and plot bars for sections of runtime on a run."""
    _, selection = __check_coherence(algorithm=algo, n=n, K=k, degree=degree)
    data = filter_data(data, selection)
    if data.shape[0] == 0:
        print(f"Skipping: {algo} {n} {k} {degree}", file=stderr)
        return
    data.replace({'section': profile_sections}, inplace=True)
    used_sections = [section for section in profile_sections.values() if section in data['section'].unique()]
    file_title = __nice_pdf_name(title)
    dest = PLOTS_DIR / branch / "profiles"
    dest.mkdir(parents=True, exist_ok=True)
    bar_plot(dest / file_title, data, 'time', __nice_lang_name(title), 'section', "[cycles]", used_sections)
