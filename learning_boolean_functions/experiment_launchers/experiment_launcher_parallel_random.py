import os
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run the tests for the random sampling approach')
parser.add_argument('dataset', help='This can be either \'crimes\' or \'superconduct\'')
parser.add_argument('--notrees', type=int, default=20)
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--ncores', type=int, default=10)
parser.add_argument('--wait_time', type=int, default=100)
args = parser.parse_args()
dataset, no_trees, dry_run, n_cores, wait_time= args.dataset, args.notrees, args.dryrun, args.ncores, args.wait_time
assert(dataset in ["crimes", "superconduct"])
if dataset == "crimes":
    n=500
elif dataset=="superconduct":
    n=324
depth_to_mem = {2: 1000, 3: 1000, 4: 1000, 5: 1000}
depth_to_time = {2: "3:59", 3: "23:59", 4: "23:59", 5: "23:59"}
depth_to_wait_time= {2:100, 3:1000, 4:10000, 5:20000}
for depth in [3, 4]:
    if depth == 3:
        C_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if depth == 4:
        C_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.4, 0.6]

    for C in C_list:
        for ratio in [1.1, 1.6, 3.0]:
            for sampling_factor in list(np.linspace(0.2, 2.0, 10)) + list(np.linspace(2.0, 6.0, 10)):
                    path = Path(f"../results/random_parallel_new/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
                                f"C={C:.3}_ratio={ratio:.3}_samplefactor={sampling_factor:.3}_ncores={n_cores}.json")
                    if not path.is_file():
                        submit_string = f"bsub -W {depth_to_time[depth]} -n {n_cores}"\
                                        f" -o logs/random_parallel/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_" \
                                        f"C={C:.3}_ratio={ratio:.3}_samplefactor={sampling_factor:.3}.txt"\
                                        f" -R rusage[mem={depth_to_mem[depth]}] "\
                                        f"python -u parallel_random_runner.py {n} {no_trees} {depth} {C} {ratio} {sampling_factor} {n_cores} {depth_to_wait_time[depth]} {dataset} "\
                                        f"&> /dev/null"
                        if not dry_run:
                            os.system(submit_string)
                        print(submit_string)




