import os
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run the tests for the proximal (CS) approach')
parser.add_argument('dataset', help='This can be either \'crimes\' or \'superconduct\'')
parser.add_argument('-n', type=int, default=20)
parser.add_argument('--notrees', type=int, default=100)
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()
dataset, n, no_trees, dry_run = args.dataset, args.n, args.notrees, args.dryrun
depth_to_mem = {2: 4000, 3: 4000, 4: 4000, 5: 10000, 6:40000, 7:40000, 8:40000}
depth_to_time = {2: "3:59", 3: "3:59", 4: "3:59", 5: "23:59", 6: "23:59", 7: "23:59", 8: "123:59"}
for depth in range(2,8):
    for C in np.linspace(0.8,1.6,10):
        for lmda_i, lmda in enumerate(10 ** np.linspace(-4,1,8)):
            for try_number in range(10):
                path = Path(f"../results/cs/{dataset}_n={n}_no_trees={no_trees}_"
                            f"C={C:.3}_lambda={lmda:.6}_tryno={try_number}.json", 'w', encoding='utf-8')
                if not path.is_file():
                    submit_string = f"bsub -W {depth_to_time[depth]} "\
                                    f" -o logs/{dataset}_n={n}_no_trees={no_trees}_C={C:.3}_lambda={lmda_i}_tryno={try_number}.txt"\
                                    f" -R rusage[mem={depth_to_mem[depth]}] "\
                                    f"python -u cs_runner.py {n} {no_trees} {depth} {try_number} {C} {lmda} {dataset} "\
                                    f"&> /dev/null"
                    if not dry_run:
                        os.system(submit_string)
                    print(submit_string)
