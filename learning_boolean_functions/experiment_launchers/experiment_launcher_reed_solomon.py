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
depth_to_mem = {2: 1000, 3: 1000, 4: 4000, 5: 4000, 6:4000, 7:10000, 8:10000}
depth_to_time = {2: "3:59", 3: "3:59", 4: "3:59", 5: "23:59", 6: "23:59", 7: "23:59", 8: "23:59"}
for depth in range(2, 6):
    for C in [0.05, 0.1, 0.15, 0.2, 0.4]:
        for ratio in [1.1, 2.6, 5.0]:
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




