import os
from pathlib import Path
import numpy as np
import argparse
from math import ceil
parser = argparse.ArgumentParser(description='Run the tests for the random binning approach')
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
depth_to_mem = {2: 1000, 3: 1000, 4: 4000, 5: 4000, 6:4000, 7:10000, 8:10000}
depth_to_time = {2: "3:59", 3: "3:59", 4: "3:59", 5: "23:59", 6: "23:59", 7: "23:59", 8: "23:59"}
C_to_ratio = {0.05: [1.1, 2.6, 5], 0.1: [1.1, 2.6, 5], 0.15: [1.1, 2.6, 5], 0.2: [1.1, 2.6, 5], 0.4: [1.1, 2.6, 5], 0.6: [2.6, 5], 0.8: [2.6, 5]}
for depth in range(2, 6):
    for C in [0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for ratio in C_to_ratio[C]:
            cs_bins_range = np.linspace(0.2 * depth ** 2 , depth ** 2, 10)
            cs_bins_range = [ceil(a) for a in cs_bins_range]
            for cs_bins in cs_bins_range:
                for cs_iterations in [1, 2, 3]:
                    for cs_ratio in [1.1, 1.3, 1.5, 1.9, 2.1, 3.0]:
                        path = Path(f"../results/binning/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
                                    f"C={C:.3}_ratio={ratio:.3}_csbins={cs_bins}_csiterations={cs_iterations}_"
                                    f"csratio={cs_ratio}.json")
                        if not path.is_file():
                            submit_string = f"bsub -W {depth_to_time[depth]} "\
                                            f" -o logs/binning/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_" \
                                            f"C={C:.3}_ratio={ratio:.3}_csbins={cs_bins}_csiterations={cs_iterations}_"\
                                            f"csratio={cs_ratio}.txt " \
                                            f"-R rusage[mem={depth_to_mem[depth]}] "\
                                            f"python -u binning_runner.py {n} {no_trees} {depth} {C} {ratio} {cs_bins} {cs_iterations} {cs_ratio} {dataset} "\
                                            f"&> /dev/null"
                            if not dry_run:
                                os.system(submit_string)
                            print(submit_string)