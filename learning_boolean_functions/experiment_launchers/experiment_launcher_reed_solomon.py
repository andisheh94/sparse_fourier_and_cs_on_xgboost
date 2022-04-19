import os
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run the tests for the reed_solomon approach')
parser.add_argument('dataset', help='This can be either \'crimes\' or \'superconduct\'')
parser.add_argument('--notrees', type=int, default=20)
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()
dataset, no_trees, dry_run = args.dataset, args.notrees, args.dryrun

if dataset == "crimes":
    n=500
elif dataset=="superconduct":
    n=324
depth_to_mem = {2: 4000, 3: 4000, 4: 10000, 5: 40000, 6:40000, 7:40000, 8:80000}
depth_to_time = {2: "3:59", 3: "3:59", 4: "3:59", 5: "23:59", 6: "23:59", 7: "23:59", 8: "123:59"}
for depth in range(2,8):
    for C in np.linspace(1, 1.8, 5):
        for ratio in np.linspace(1.1, 2.1, 5):
            path = Path(f"../results/reed_solomon/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
                        f"C={C:.3}_ratio={ratio:.3}.json")
            if not path.is_file():
                submit_string = f"bsub -W {depth_to_time[depth]} "\
                                f" -o logs/reed_solomon/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_" \
                                f"C={C:.3}_ratio={ratio:.3}.txt"\
                                f" -R rusage[mem={depth_to_mem[depth]}] "\
                                f"python -u reed_solomon_runner.py {n} {no_trees} {depth} {C} {ratio} {dataset} "\
                                f"&> /dev/null"
                if not dry_run:
                    os.system(submit_string)
                print(submit_string)




