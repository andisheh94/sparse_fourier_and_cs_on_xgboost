from random_forest_builder.random_forest_model import RandomForestModel, Fourier
from sparse_wht_algorithms.swht.build.src.python_module.swht import swht, REED_SOLOMON
import sys
import numpy as np
import time
import json
if __name__ == "__main__":
    n, no_trees, depth, try_number = [int(sys.argv[j]) for j in [1, 2, 3, 4]]
    C, ratio = [float(sys.argv[j]) for j in [5, 6]]
    dataset = sys.argv[7]
    np.random.seed(try_number)
    random_forest_model = RandomForestModel(dataset, n, no_trees, depth)
    true_fourier_transform = random_forest_model.get_fourier_transform()
    k = true_fourier_transform.get_sparsity()

    # Get Fourier transform
    # Fill cache
    fourier_transform = swht(random_forest_model.model, REED_SOLOMON, n, k, C=C, degree=depth)
    # Reset sampling complexity after filling cache
    random_forest_model.reset_sampling_complexity()
    # Get Fourier transform again
    start = time.time()
    fourier_transform = swht(random_forest_model.model, REED_SOLOMON, n, k, C=C, degree=depth)
    end = time.time()
    elapsed_time = end - start
    print(fourier_transform)
    fourier_transform = Fourier.from_tuple_series(fourier_transform)
    equality = (fourier_transform == true_fourier_transform)
    mse = Fourier.get_mse(fourier_transform, true_fourier_transform)
    true_fourier_norm_squared, computed_fourier_norm_squared = true_fourier_transform.norm_squared(), fourier_transform.norm_squared()
    with open(f"../results/reed_solomon/{dataset}_n={n}_no_trees={no_trees}_depth={depth}_"
              f"C={C:.3}_ratio={ratio:.3}_tryno={try_number}.json", 'w', encoding='utf-8') as f:
        results_dict = {"n": n, "no_trees": no_trees, "depth": depth, "C": C, "ratio": ratio,
                        "try_number": try_number, "k": k, "time": elapsed_time, "equality": equality, "mse": mse,
                        "true_fourier_norm_squared": true_fourier_norm_squared,
                        "computed_fourier_norm_squared": computed_fourier_norm_squared,
                        "measurements": random_forest_model.get_sampling_complexity()}
        print(results_dict)
        json.dump(results_dict, f)








