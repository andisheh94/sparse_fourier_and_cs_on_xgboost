"""Full Benchmark
Runs a selection of parameters through the benchmark as individual jobs and
records their status.
"""


def main():
    from numpy import linspace, uint64
    from pathlib import Path
    from subprocess import run as sysrun, CalledProcessError
    from re import compile as re_compile
    from sys import stderr
    from argparse import ArgumentParser

    ALL_CS = ("naive", "randbin", "reedsolo")

    # Arguments
    parser = ArgumentParser("full_benchmark")
    parser.add_argument("--profile", action="store_true", dest="do_profile", help="Run profiling targets")
    parser.add_argument("-p", action="store_true", dest="use_python", help="Time with the python interface")
    parser.add_argument("-b", action="store_true", dest="use_basic", help="Time the basic version only")
    parser.add_argument("-r", action="store_true", dest="use_robust", help="Time the robust version only")
    parser.add_argument("cs_algo", type=str, nargs='*', action="store", default=[], help="Time a specific CS algorithm")
    args = parser.parse_args()
    for algo in args.cs_algo:
        if algo not in ALL_CS:
            raise ValueError(f"Unrecognized CS algorithm: {algo}")

    # Working variables
    interface = "py" if args.use_python else "raw"
    project_root_path = Path(__file__).resolve().parent.parent
    benchmark_path = project_root_path / "benchmarking"
    if args.do_profile:
        benchmark_out_path = benchmark_path / "profiles" / "out"
    else:
        benchmark_out_path = benchmark_path / "results" / "out"
    benchmark_out_path.mkdir(parents=True, exist_ok=True)
    benchmark_exec_path = project_root_path / "build" / "benchmarking"
    pattern = re_compile(r"Job <(\d+)>")

    # Parameters
    all_robustness = []
    if args.use_basic:
        all_robustness.append(('basic', ''))
    if args.use_robust:
        all_robustness.append(('robust', '-r '))
    if not all_robustness:
        all_robustness = [('basic', ''), ('robust', '-r ')]
    all_cs = args.cs_algo if args.cs_algo else ALL_CS
    all_degree = [1, 2, 5, 10]
    all_n = linspace(10, 10000, 5).round().astype(uint64)
    all_K = linspace(10, 300, 4).round().astype(uint64)
    runtime = "12:00"
    runs_per_n_d = len(all_robustness) * len(all_cs) * all_K.size
    total_runs = runs_per_n_d * all_n.size * len(all_degree)
    for n in all_n:
        if n not in all_degree:
            total_runs += runs_per_n_d

    # Safety compilation
    print("Compiling...")
    sysrun(["./setup.sh", "ready", "-D", "Profile" if args.do_profile else "Benchmark"],
        cwd=str(project_root_path), check=True
    )

    # Run
    print("Running...")
    job_names = []
    job_ids = []
    i = 1
    print(f"0 / {total_runs}", end='', flush=True)
    for robustness, r_flag in all_robustness:
        for cs in all_cs:
            for k in all_K:
                for n in all_n:
                    for degree in all_degree + ([n] if n not in all_degree else []):
                        try:
                            job_name = f"{interface}_{robustness}_{cs}_{n}_{k}_{degree}"
                            out = sysrun(
                                (f"bsub -R 'select[model==XeonGold_5118]' -n 12 -W {runtime} -oo {benchmark_out_path / job_name}"
                                f" -J {project_root_path.name}_{job_name} -G ls_krausea ./{interface}_timing {r_flag}{cs} {n} {k} {degree}"),
                                shell=True, cwd=str(benchmark_exec_path), capture_output=True, check=True, text=True)
                            job_ids.append(next(pattern.finditer(out.stdout))[1])
                            job_names.append(job_name)
                            print(f"\r{i} / {total_runs}", end='', flush=True)
                            i += 1
                        except (CalledProcessError, KeyboardInterrupt):
                            print(f"\nAn error happened with {robustness} {cs} n={n} K={k} d={degree}, interrupting jobs", file=stderr)
                            if job_ids:
                                sysrun(["bkill"] + job_ids, shell=True, check=True)
                            raise
    print(f"\n{total_runs} jobs were started.")
    with open(benchmark_path / "jobs.txt", 'w') as report:
        report.writelines([f"<{jid}> - {jname}\n" for jid, jname in zip(job_ids, job_names)])


# Script runner
if __name__ == "__main__":
    main()
