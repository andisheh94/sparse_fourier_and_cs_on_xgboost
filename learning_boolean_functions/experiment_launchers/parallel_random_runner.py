from random_forest_builder.random_forest_model import RandomForestModel, Fourier
from sparse_wht_algorithms.swht_python.sparseWHT_parallel import SWHTRobust
import sys
import numpy as np
import time
import json
if __name__ == "__main__":
    n, no_trees, depth, n_cores, wait_time = [int(sys.argv[j]) for j in [1, 2, 3, 7, 8]]
    C, ratio, sampling_factor = [float(sys.argv[j]) for j in [4, 5, 6]]
    dataset = sys.argv[9]
    np.random.seed(0)
    random_forest_model = RandomForestModel(dataset, n, no_trees, depth)
    true_fourier_transform = random_forest_model.get_fourier_transform()
    k = true_fourier_transform.get_sparsity()
    # Get Fourier transform
    start = time.time()
    SWHTRobust(n, k, finite_field_class="random_cs", C=C, ratio=ratio, degree=depth,
               sampling_factor=sampling_factor, no_processes=n_cores,
               wait_time=wait_time).run(random_forest_model, seed=0)
    end = time.time()
    elapsed_time_uncached = end - start
    random_forest_model.reset_sampling_complexity()
    # Get Fourier transform again cached
    start = time.time()
    fourier_transform = SWHTRobust(n, k, finite_field_class="random_cs", C=C, ratio=ratio, degree=depth,
                                   sampling_factor=sampling_factor, no_processes=n_cores,
                                   wait_time=wait_time).run(random_forest_model, seed=0)
    end = time.time()
    elapsed_time_cached = end - start
    fourier_transform = Fourier.from_tuple_series(fourier_transform)
    equality = (fourier_transform == true_fourier_transform)
    mse = Fourier.get_mse(fourier_transform, true_fourier_transform)
    true_fourier_norm_squared, computed_fourier_norm_squared = true_fourier_transform.norm_squared(), fourier_transform.norm_squared()
    with open(f"../results/random_parallel_new/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
              f"C={C:.3}_ratio={ratio:.3}_samplefactor={sampling_factor:.3}_ncores={n_cores}.json", 'w', encoding='utf-8') as f:
        results_dict = {"n": n, "no_trees": no_trees, "depth": depth, "C": C, "ratio":ratio,
                        "sampling_factor": sampling_factor, "n_cores": n_cores, "wait_time": wait_time, "k": k,
                        "time_uncached": elapsed_time_uncached, "time_cached": elapsed_time_cached,
                        "equality": equality, "mse": mse,
                        "true_fourier_norm_squared": true_fourier_norm_squared,
                        "computed_fourier_norm_squared": computed_fourier_norm_squared,
                        "measurements": random_forest_model.get_sampling_complexity()}
        print(results_dict)
        json.dump(results_dict, f)