"""
File management utilities
-------------------------

Defines functions to filter and merge benchmarking results into manageable
pandas-based csv files.
"""

from pathlib import Path
from re import compile as regex_compile
from datetime import datetime
from typing import List, Tuple
from itertools import product as cartesian_product
from pandas import read_csv, concat
from pandas.core.frame import DataFrame
from shutil import copytree, rmtree

from ._constants import options_types, RESULTS_DIR, RESULTS_SOURCES, ALGO_MAP


def results_path(branch: str, sectioned: bool = False) -> Path:
    """Get path to results files."""
    return RESULTS_DIR / branch / ("profiles" if sectioned else "results")


def load_sources(branch: str, purge: bool = True) -> Path:
    """Refresh result files for a branch."""
    branch_results = RESULTS_DIR / branch
    branch_source = RESULTS_SOURCES / branch
    if purge:
        rmtree(branch_results, ignore_errors=True)
    copytree(branch_source, branch_results)
    return branch_results


def read_results_file(file: Path, indexed: bool = True) -> DataFrame:
    index_column = 0 if indexed else None
    try:
        return read_csv(file, header=0, index_col=index_column, dtype=options_types, float_precision='high')
    except ValueError as e:
        print(file)
        raise e


def merge_results(path: Path, result_files: List[Path]):
    """Combine results files by interface and robustness into a single csv.
    """

    # Read and sort all the results as dataframes
    results = {index: [] for index in cartesian_product(('raw', 'py', 'old'), ('basic', 'robust'))}
    for result_file in result_files:
        interface, robustness, _ = result_file.stem.split('_', 2)
        results[(interface, robustness)].append(
            read_results_file(result_file, indexed=False)
        )
        result_file.unlink()

    # Combine and write each main dataframe
    for (interface, robustness), data in results.items():

        # Merge the dataframes
        if not data:
            continue
        dataframe: DataFrame = concat(data, ignore_index=True, copy=False)

        # Systematize the columns
        dataframe['algorithm'].replace(ALGO_MAP, inplace=True)

        # Combine with pre-existing results
        target_file = path / f"{interface}_{robustness}.csv"
        if target_file.is_file():
            dataframe = concat((
                    read_results_file(target_file),
                    dataframe
                ), ignore_index=True, copy=False)

        # Write to file
        dataframe.to_csv(target_file)


def filter_results(results_path: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """Organize results by logging failed runs and
    storing partial results in corresponding directories.
    """

    # Ready directories
    out_path = results_path / "out"
    timeout_path = results_path / "timeouts"
    timeout_path.mkdir(exist_ok=True)
    failures_path = results_path / "failures"
    failures_path.mkdir(exist_ok=True)

    # Initialize matching
    timeout_pattern = regex_compile(r"TERM_RUNLIMIT")
    success_pattern = regex_compile(r"Successfully completed.")
    date_format = "%a %b %d %H:%M:%S %Y"
    str_to_date = lambda str_date: datetime.strptime(str_date, date_format)
    bench_start = datetime.max
    bench_end = datetime.min
    failure_encountered = False
    
    # Records
    successful_results = []
    timeout_results = []
    failure_results = []

    # Parse output files to check result status
    for outfile in out_path.iterdir():

        # Get completion time info
        with open(outfile, 'r') as outfile_body:
            for _ in range(7): outfile_body.readline()
            experiment_start = outfile_body.readline().strip()[11:]
            experiment_end = outfile_body.readline().strip()[14:]
            text_body = outfile_body.read()
        bench_start = min(bench_start, str_to_date(experiment_start))
        bench_end = max(bench_end, str_to_date(experiment_end))

        # Locate result file
        job_name = outfile.stem
        result_file = results_path / f"{job_name}.csv"

        # Store good results for merging
        if success_pattern.search(text_body):
            successful_results.append(result_file)
            
        # Filter and move bad results
        else:
            if timeout_pattern.search(text_body):
                target_path = timeout_path
                target_aggregator = timeout_results
            else:
                failure_encountered = True
                target_path = failures_path
                target_aggregator = failure_results

            # If a partial result exists, move it
            if result_file.is_file():
                with open(result_file, 'r') as target_body:
                    n_lines = len(target_body.readlines())
                if n_lines > 1:
                    new_path = target_path / result_file.name
                    result_file.rename(new_path)
                    target_aggregator.append(new_path)
                else:
                    result_file.unlink()
            
            # Log the parameters of the failed run
            with open(target_path / "record.csv", 'a') as record:
                record.write(f"{job_name.replace('_', ',')}\n")
        
        # Clear output file
        outfile.unlink()
    
    # Give feedback and successes
    print(f"Total runtime: {bench_end - bench_start}")
    out_path.rmdir()
    if failure_encountered:
        print("A failure case was encountered.")
    return successful_results, timeout_results, failure_results
