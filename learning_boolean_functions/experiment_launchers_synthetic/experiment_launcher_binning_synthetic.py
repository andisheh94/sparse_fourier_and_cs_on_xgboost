import os
from pathlib import Path
import numpy as np
import argparse
from math import ceil
parser = argparse.ArgumentParser(description='Run the tests for the random binning approach on a synthetic dataset')
parser.add_argument('-n', type=int, default=500)
parser.add_argument('-k', type=int, default=30)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('--dryrun', action='store_true')

args = parser.parse_args()
n, k, seed, dry_run = args.n, args.k, args.seed, args.dryrun
for degree in [2, 3, 4, 5, 10, 20]:
    print(degree)
    for C in [0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0,  2.0, 3.0]:
        for ratio in [1.1, 1.6, 3.0]:
            cs_bins_range = np.linspace(0.2 * degree ** 2 if degree < 10 else 0.05 * degree ** 2,
                                        degree ** 2 if degree < 10 else 0.5 * degree ** 2, 10)
            cs_bins_range = [ceil(a) for a in cs_bins_range]
            for cs_bins in cs_bins_range:
                for cs_iterations in [1, 2, 3]:
                    for cs_ratio in [1.1, 1.3, 1.5, 1.9, 2.1, 3.0]:
                        path = Path(f"../results_synthetic/binning/n={n}_k={k}_degree={degree}_"
                                    f"C={C:.3}_ratio={ratio:.3}_csbins={cs_bins}_csiterations={cs_iterations}_"
                                    f"csratio={cs_ratio}_seed={seed}.json")
                        if not path.is_file():
                            submit_string = f"bsub -W 3:59 "\
                                            f" -o logs/binning_synthetic/n={n}_k={k}_degree={degree}_" \
                                            f"C={C:.3}_ratio={ratio:.3}_csbins={cs_bins}_csiterations={cs_iterations}_"\
                                            f"csratio={cs_ratio}_seed={seed}.txt " \
                                            f"-R rusage[mem=2000] "\
                                            f"python -u binning_runner_synthetic.py {n} {k} {degree} {C} {ratio} {cs_bins} {cs_iterations} {cs_ratio} {seed} "\
                                            f"&> /dev/null"
                            if not dry_run:
                                os.system(submit_string)
                            print(submit_string)