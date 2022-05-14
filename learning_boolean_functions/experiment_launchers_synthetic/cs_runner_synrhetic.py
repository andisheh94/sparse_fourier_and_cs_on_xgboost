from random_forest_builder.random_forest_model import RandomForestModel, Fourier
from sparse_wht_algorithms.compressive_sensing.proximal_method import ProximalMethod
import sys
import numpy as np
import time
import json
if __name__ == "__main__":
    n, no_trees, depth = [int(sys.argv[j]) for j in [1, 2, 3]]
    C, lmda = [float(sys.argv[j]) for j in [4, 5]]
    dataset = sys.argv[6]
    np.random.seed(0)
    random_forest_model = RandomForestModel(dataset, n, no_trees, depth)
    true_fourier_transform = random_forest_model.get_fourier_transform()
    k = true_fourier_transform.get_sparsity()
    swht_proximal = ProximalMethod(n, k, depth, C)
    # Get Fourier transform
    start = time.time()
    swht_proximal.sample_function(true_fourier_transform)
    end = time.time()
    sampling_time = end - start
    start = time.time()
    fourier_transform = Fourier.from_tuple_series(swht_proximal.run(lmda))
    end = time.time()
    opt_time = end - start
    equality = (fourier_transform == true_fourier_transform)
    mse = Fourier.get_mse(fourier_transform, true_fourier_transform)
    true_fourier_norm_squared, computed_fourier_norm_squared = true_fourier_transform.norm_squared(), fourier_transform.norm_squared()
    with open(f"../results/cs/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
              f"C={C:.3}_lambda={lmda:.6}.json", 'w', encoding='utf-8') as f:
        results_dict = {"n": n, "no_trees": no_trees, "depth": depth, "C": C, "lambda": lmda,
                        "k": k, "sampling_time": sampling_time,
                        "optimization_time": opt_time,
                        "equality": equality, "mse": mse,
                        "true_fourier_norm_squared": true_fourier_norm_squared,
                        "computed_fourier_norm_squared": computed_fourier_norm_squared,
                        "measurements": swht_proximal.get_number_of_measurements()}
        print(results_dict)
        json.dump(results_dict, f)








