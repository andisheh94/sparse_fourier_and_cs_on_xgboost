import os
from pathlib import Path
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Run the tests for the random  measruements approach on a synthetic dataset')
parser.add_argument('-n', type=int, default=500)
parser.add_argument('-k', type=int, default=30)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--ncores', type=int, default=10)
parser.add_argument('--wait_time', type=int, default=100)
args = parser.parse_args()
n, k, seed, dry_run, n_cores, wait_time = args.n, args.k, args.seed, args.dryrun, args.ncores, args.wait_time
deg_to_time = {2: "3:59", 3: "23:59", 4: "123:59", 5: "123:59"}
deg_to_mem = {2: 4000, 3: 10000, 4: 20000, 5: 40000}
for degree in [4]:
    print(degree)
    if degree==4:
        wait_time = 1000
    for C in [0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0]:
        for ratio in [1.1, 1.6, 3.0]:
            for sampling_factor in np.linspace(2.0, 6.0, 10):
                    path = Path(f"../results_synthetic/random_parallel/n={n}_k={k}_degree={degree}_"
                    f"C={C:.3}_ratio={ratio:.3}_samplefactor={sampling_factor:.3}_seed={seed}.json")
                    if not path.is_file():
                        submit_string = f"bsub -W {deg_to_time[degree]} -n {n_cores}"\
                                        f" -o logs/random_parallel_synthetic/n={n}_k={k}_degree={degree}_" \
                                        f"C={C:.3}_ratio={ratio:.3}_samplefactor={sampling_factor:.3}_seed={seed}.txt " \
                                        f"-R rusage[mem=20000] "\
                                        f"python -u parallel_random_runner_synthetic.py  {n}  {k}  {degree} {C} {ratio} {sampling_factor} {n_cores} {wait_time} {seed} "\
                                        f"&> /dev/null"
                        if not dry_run:
                            os.system(submit_string)
                        print(submit_string)




