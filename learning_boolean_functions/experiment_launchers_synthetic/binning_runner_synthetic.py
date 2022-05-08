from sparse_wht_algorithms.swht_python.utils.random_function import RandomFunction
from sparse_wht_algorithms.swht_python.sparseWHT_robust_sampleoptimal import SWHTRobust
from random_forest_builder.fourier import Fourier
import sys
import numpy as np
import time
import json
if __name__ == "__main__":
    "n, k, degree, C, ratio, cs_bins, cs_iterations, cs_ratio, seed"
    n, k, degree, cs_bins, cs_iterations, seed = [int(sys.argv[j]) for j in [1, 2, 3, 6, 7, 9]]
    C, ratio, cs_ratio= [float(sys.argv[j]) for j in [4, 5, 8]]
    random_f = RandomFunction(n, k, degree, seed)
    true_fourier_transform = random_f.get_fourier_transform()
    k = true_fourier_transform.get_sparsity()
    # print(true_fourier_transform)
    # Get Fourier transform
    start = time.time()
    SWHTRobust(n, k, C=C, ratio=ratio, finite_field_class="binary_search_cs", cs_bins=cs_bins, cs_iterations=cs_iterations,
                                   cs_ratio=cs_ratio).run(random_f, seed=seed)
    end = time.time()
    elapsed_time_uncached = end - start
    # Get fourier transform this time cached
    random_f.use_cache = True
    random_f.reset_sampling_complexity()
    start = time.time()
    fourier_transform_tuple = SWHTRobust(n, k, C=C, ratio=ratio, finite_field_class="binary_search_cs", cs_bins=cs_bins, cs_iterations=cs_iterations,
                                   cs_ratio=cs_ratio).run(random_f, seed=seed)
    end = time.time()
    elapsed_time_cached = end - start
    fourier_transform = Fourier.from_tuple_series(fourier_transform_tuple)
    equality = (fourier_transform == true_fourier_transform)
    mse = Fourier.get_mse(fourier_transform, true_fourier_transform)
    true_fourier_norm_squared, computed_fourier_norm_squared = true_fourier_transform.norm_squared(), fourier_transform.norm_squared()
    with open(f"../results_synthetic/binning/n={n}_k={k}_degree={degree}_"
              f"C={C:.3}_ratio={ratio:.3}_csbins={cs_bins}_csiterations={cs_iterations}_"
              f"csratio={cs_ratio}_seed={seed}.json", 'w', encoding='utf-8') as f:
        results_dict = {"n": n, "k": k, "degree": degree , "C": C, "ratio": ratio,
                        "cs_bins": cs_bins, "cs_iteraions": cs_iterations,  "cs_ratio": cs_ratio,
                        "time_uncached": elapsed_time_uncached,
                        "time_cached": elapsed_time_cached, "equality": equality, "mse": mse,
                        "true_fourier_norm_squared": true_fourier_norm_squared,
                        "computed_fourier_norm_squared": computed_fourier_norm_squared,
                        "measurements": random_f.get_sampling_complexity()}
        print(results_dict)
        json.dump(results_dict, f)



