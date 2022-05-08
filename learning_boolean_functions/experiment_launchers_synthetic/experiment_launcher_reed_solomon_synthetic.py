import os
from pathlib import Path
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Run the tests for the reed solomon approach on a synthetic dataset')
parser.add_argument('-n', type=int, default=500)
parser.add_argument('-k', type=int, default=30)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()
n, k, seed, dry_run = args.n, args.k, args.seed, args.dryrun

deg_to_mem = {2: 1000, 3: 1000, 4: 4000, 5: 4000, 10:10000, 20:10000}
deg_to_time = {2: "3:59", 3: "3:59", 4: "3:59", 5: "3:59", 10: "23:59", 20: "23:59"}
for degree in [2, 3, 4, 5, 10, 20]:
    for C in [0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0]:
        for ratio in [1.1, 1.6, 3.0]:
            path = Path(f"../results_synthetic/reed_solomon_synthetic/n={n}_k={k}_degree={degree}_"
                        f"C={C:.3}_ratio={ratio:.3}.json")
            if not path.is_file():
                submit_string = f"bsub -W {deg_to_time[degree]}"\
                                f" -o logs/reed_solomon_synthetic/n={n}_k={k}_degree={degree}_" \
                                f"C={C:.3}_ratio={ratio:.3}" \
                                f" -R rusage[mem={deg_to_mem[degree]}] "\
                                f"python -u reed_solomon_runner_synthetic.py {n} {k} {degree} {C} {ratio} {seed} "\
                                f"&> /dev/null"
                if not dry_run:
                    os.system(submit_string)
                print(submit_string)




