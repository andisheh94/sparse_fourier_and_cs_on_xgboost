import os
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run the tests for the proximal (CS) approach')
parser.add_argument('dataset', help='This can be either \'crimes\' or \'superconduct\'')
parser.add_argument('--notrees', type=int, default=20)
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()
dataset, no_trees, dry_run = args.dataset, args.notrees, args.dryrun
assert(dataset in ["crimes", "superconduct"])
if dataset == "crimes":
    n=500
elif dataset=="superconduct":
    n=324
depth_to_mem = {2: 4000, 3: 80000}
depth_to_time = {2: "3:59", 3: "23:59"}
for depth in range(2,5):
    for C in np.linspace(0.1, 1.4, 10):
        for lmda_i, lmda in enumerate(10 ** np.linspace(-5,2,8)):
            path = Path(f"../results/cs/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
                        f"C={C:.3}_lambda={lmda:.6}.json", 'w', encoding='utf-8')
            if not path.is_file():
                submit_string = f"bsub -W {depth_to_time[depth]} "\
                                f" -o logs/cs/{dataset}_n={n}_no_trees={no_trees}_depth={depth}.txt" \
                                f"_C={C:.3}_lambda={lmda:.6}.txt"\
                                f" -R rusage[mem={depth_to_mem[depth]}] "\
                                f"python -u cs_runner.py {n} {no_trees} {depth} {C} {lmda} {dataset} "\
                                f"&> /dev/null"
                if not dry_run:
                    os.system(submit_string)
                print(submit_string)
