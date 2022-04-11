from random_forest_builder.random_forest_model import RandomForestModel, Fourier
from sparse_wht_algorithms.swht.build.src.python_module.swht import swht, RANDOM_BINNING
import sys
import numpy as np
import time
import json
if __name__ == "__main__":
    n, no_trees, depth = [int(sys.argv[j]) for j in [1, 2, 3]]
    C, ratio, cs_bins, cs_iterations, cs_ratio = [float(sys.argv[j]) for j in [4, 5, 6, 7, 8]]
    dataset = sys.argv[9]
    # Consistency in results
    np.random.seed(0)
    random_forest_model = RandomForestModel(dataset, n, no_trees, depth)
    true_fourier_transform = random_forest_model.get_fourier_transform()
    k = true_fourier_transform.get_sparsity()

    # Get Fourier transform
    # Get Fourier transform again
    start = time.time()
    fourier_transform = swht(true_fourier_transform, RANDOM_BINNING , n, k, C=C, ratio=ratio, cs_bins=cs_bins,
                             cs_iterations=cs_iterations, cs_ratio=cs_ratio)
    end = time.time()
    elapsed_time = end - start
    fourier_transform = Fourier.from_tuple_series(fourier_transform)
    equality = (fourier_transform == true_fourier_transform)
    mse = Fourier.get_mse(fourier_transform, true_fourier_transform)
    true_fourier_norm_squared, computed_fourier_norm_squared = true_fourier_transform.norm_squared(), fourier_transform.norm_squared()
    with open(f"../results/binning/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
              f"C={C:.3}_ratio={ratio:.3}_csbins={cs_bins}_csiterations={cs_iterations}_"
              f"csratio={cs_ratio}.json", 'w', encoding='utf-8') as f:
        results_dict = {"n": n, "no_trees": no_trees, "depth": depth, "C": C, "ratio": ratio,
                        "cs_bins": cs_bins, "cs_iteraions": cs_iterations,  "cs_ratio": cs_ratio,
                        "k": k, "time": elapsed_time, "equality": equality, "mse": mse,
                        "true_fourier_norm_squared": true_fourier_norm_squared,
                        "computed_fourier_norm_squared": computed_fourier_norm_squared,
                        "measurements": true_fourier_transform.get_sampling_complexity()}
        print(results_dict)
        json.dump(results_dict, f)



