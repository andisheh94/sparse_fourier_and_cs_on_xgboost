from random_forest_builder.random_forest_model import RandomForestModel, Fourier
from sparse_wht_algorithms.swht_python.sparseWHT_robust_sampleoptimal import SWHTRobust
import sys
import numpy as np
import time
import json
if __name__ == "__main__":
    "n, no_trees, depth, C, ratio, cs_bins, cs_iterations, cs_ratio"
    n, no_trees, depth, cs_bins, cs_iterations   = [int(sys.argv[j]) for j in [1, 2, 3, 6, 7]]
    C, ratio, cs_ratio = [float(sys.argv[j]) for j in [4, 5, 8]]
    dataset = sys.argv[9]
    np.random.seed(0)
    random_forest_model = RandomForestModel(dataset, n, no_trees, depth)
    true_fourier_transform = random_forest_model.get_fourier_transform()
    k = true_fourier_transform.get_sparsity()
    # Get Fourier transform
    start = time.time()
    SWHTRobust(n, k, C=C, ratio=ratio, finite_field_class="binary_search_cs", cs_bins=cs_bins, cs_iterations=cs_iterations,
                                   cs_ratio=cs_ratio).run(random_forest_model, seed=0)
    end = time.time()
    elapsed_time_uncached = end - start
    # Get fourier transform this time cached
    random_forest_model.reset_sampling_complexity()
    start = time.time()
    fourier_transform_tuple = SWHTRobust(n, k, C=C, ratio=ratio, finite_field_class="binary_search_cs", cs_bins=cs_bins, cs_iterations=cs_iterations,
                                   cs_ratio=cs_ratio).run(random_forest_model, seed=0)
    end = time.time()
    elapsed_time_cached = end - start
    fourier_transform = Fourier.from_tuple_series(fourier_transform_tuple)
    equality = (fourier_transform == true_fourier_transform)
    mse = Fourier.get_mse(fourier_transform, true_fourier_transform)
    true_fourier_norm_squared, computed_fourier_norm_squared = true_fourier_transform.norm_squared(), fourier_transform.norm_squared()
    with open(f"../results/binning/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
              f"C={C:.3}_ratio={ratio:.3}_csbins={cs_bins}_csiterations={cs_iterations}_"
              f"csratio={cs_ratio}.json", 'w', encoding='utf-8') as f:
        results_dict = {"n": n, "no_trees": no_trees, "depth": depth, "C": C, "ratio": ratio,
                        "cs_bins": cs_bins, "cs_iteraions": cs_iterations,  "cs_ratio": cs_ratio,
                        "k": k, "time_uncached": elapsed_time_uncached,
                        "time_cached": elapsed_time_cached, "equality": equality, "mse": mse,
                        "true_fourier_norm_squared": true_fourier_norm_squared,
                        "computed_fourier_norm_squared": computed_fourier_norm_squared,
                        "measurements": random_forest_model.get_sampling_complexity()}
        print(results_dict)
        json.dump(results_dict, f)



