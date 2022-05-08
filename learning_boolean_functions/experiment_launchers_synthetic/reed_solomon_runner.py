from random_forest_builder.random_forest_model import RandomForestModel, Fourier
from sparse_wht_algorithms.swht_python.sparseWHT_robust_sampleoptimal import SWHTRobust
import sys
import numpy as np
import time
import json
if __name__ == "__main__":
    n, no_trees, depth = [int(sys.argv[j]) for j in [1, 2, 3]]
    C, ratio = [float(sys.argv[j]) for j in [4, 5]]
    dataset = sys.argv[6]
    np.random.seed(0)
    random_forest_model = RandomForestModel(dataset, n, no_trees, depth)
    true_fourier_transform = random_forest_model.get_fourier_transform()
    k = true_fourier_transform.get_sparsity()
    # Get Fourier transform
    start = time.time()
    SWHTRobust(n, k, C=C, ratio=ratio, finite_field_class="reed_solomon_cs", degree=depth).run(random_forest_model, seed=0)
    end = time.time()
    elapsed_time_uncached = end - start
    random_forest_model.reset_sampling_complexity()
    # Get Fourier transform again
    random_forest_model.use_cache = True
    start = time.time()
    fourier_transform = SWHTRobust(n, k, C=C, ratio=ratio, finite_field_class="reed_solomon_cs", degree=depth).run(
        random_forest_model, seed=0)
    end = time.time()
    elapsed_time_cached = end - start
    fourier_transform = Fourier.from_tuple_series(fourier_transform)
    equality = (fourier_transform == true_fourier_transform)
    mse = Fourier.get_mse(fourier_transform, true_fourier_transform)
    true_fourier_norm_squared, computed_fourier_norm_squared = true_fourier_transform.norm_squared(), fourier_transform.norm_squared()
    with open(f"../results/reed_solomon/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
              f"C={C:.3}_ratio={ratio:.3}.json", 'w', encoding='utf-8') as f:
        results_dict = {"n": n, "no_trees": no_trees, "depth": depth, "C": C, "ratio": ratio,
                        "k": k, "time_uncached": elapsed_time_uncached,
                        "time_cached": elapsed_time_cached, "equality": equality, "mse": mse,
                        "true_fourier_norm_squared": true_fourier_norm_squared,
                        "computed_fourier_norm_squared": computed_fourier_norm_squared,
                        "measurements": random_forest_model.get_sampling_complexity()}
        print(results_dict)
        json.dump(results_dict, f)