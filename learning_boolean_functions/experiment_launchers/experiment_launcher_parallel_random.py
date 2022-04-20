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
depth_to_mem = {2: 4000, 3: 4000, 4: 4000, 5: 4000}
depth_to_time = {2: "3:59", 3: "3:59", 4: "23:59", 5: "23:59"}
for depth in range(2, 6):
    for C in np.linspace(0.6, 1.6, 6):
        for ratio in np.linspace(1.1, 2.1, 5):
            for sampling_factor in np.linspace(0.2,1.4,10):
                    path = Path(f"../results/random_parallel/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
                                f"C={C:.3}_ratio={ratio:.3}_samplefactor={sampling_factor:.3}.json")
                    if not path.is_file():
                        submit_string = f"bsub -W {depth_to_time[depth]} "\
                                        f" -o logs/random_parallel/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_" \
                                        f"C={C:.3}_ratio={ratio:.3}_samplefactor={sampling_factor:.3}.txt"\
                                        f" -R rusage[mem={depth_to_mem[depth]}] "\
                                        f"python -u parallel_random_runner.py {n} {no_trees} {depth} {C} {ratio} {sampling_factor} {n_cores} {wait_time} {dataset} "\
                                        f"&> /dev/null"
                        if not dry_run:
                            os.system(submit_string)
                        print(submit_string)




