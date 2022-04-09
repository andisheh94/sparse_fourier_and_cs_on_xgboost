import apgpy as apg
import numpy as np
from math import log2, isclose
from helpers.random_function import RandomFunction
from random_forest_builder.fourier import Fourier, TOLERANCE_ZERO
from itertools import combinations
from tqdm import tqdm
class ProximalMethod():
    def __init__(self, n, k, degree, C):
        # Dimension
        self.n = n
        # Sparsity
        self.k = k
        # Sampling constant factor
        self.C = C
        # Degree
        self.degree = degree
        # m is number of measurements
        self.m = int(self.C * self.k * self.degree * log2(self.n))
        # Boolean flag
        self.function_is_sampled = False
        self.freqlist = {}
        self.no_basis_functions = 0
        coordinates = range(self.n)
        for d in range(self.degree+1):
            frequencies = list(combinations(coordinates, d))
            # print(frequencies)
            for freq in frequencies:
                freq = list(freq)
                temp = np.zeros(self.n, dtype=int)
                temp[freq] = 1
                self.freqlist[self.no_basis_functions] = list(temp)
                self.no_basis_functions += 1
        # Measurement matrix
        self.psi = np.zeros((self.m, self.no_basis_functions))
        # Linear measurements vector
        self.y = np.zeros(self.m)


    def get_degree(self, freq):
        freq = self.to_binary(freq)
        return np.sum(freq)
    # def load_graph(self):
    #     g = DataGraph(0, {})
    #     with open('GraphData/1.pkl', 'rb') as f:
    #         d = pickle.load(f)
    #         g.graph = d["graph"]
    #         g.n_v = d["n_v"]
    #         g.n_e = d["n_e"]
    #         g.shape = d["shape"]
    #     return g

    def to_binary(self, i):
        # Converts integer i into an (n,1) 0-1 vector
        a = list(bin(i)[2:].zfill(self.n))
        a = [int(x) for x in a]
        # ca = np.array(a, dtype=np.intc)
        return a

    def grad(self, x):
        # grad = 2 \psi^T (\psi x - y)
        # print(self.psi, x)
        # input()
        out = np.dot(self.psi.transpose(), (np.dot(self.psi, x) - self.y))
        return out/(self.lmda*self.m)

    def proximal(self, v, t):
        return np.sign(v) * np.maximum(abs(v) - t, 0)

    def sample_function(self, f):
        t_dict = set()  # Avoid redundant t
        print(f"Generating {self.m} samples")
        for j in tqdm(range(self.m)):
            # Generate a random input
            t = [np.random.randint(0, 2) for _ in range(self.n)]
            while tuple(t) in t_dict:
                t = [np.random.randint(0, 2) for _ in range(self.n)]
            t = tuple(t)
            t_dict.add(t)
            for freq in range(self.no_basis_functions):
                self.psi[j, freq] = (-1) ** np.dot(t, self.freqlist[freq])
            self.y[j] = f.__getitem__(t)
        self.function_is_sampled = True
            # Subtract off estimate
        # print("sampling finished")
        # print(self.psi, self.y)

    def run(self, lmda=0.01):
        if self.function_is_sampled is False:
            print("Need to sample function first")
            return Fourier({})
        self.lmda = lmda
        x = apg.solve(self.grad, self.proximal, np.zeros(self.no_basis_functions), quiet=True)
        print(x)
        return self._get_fourier_series_from_vector(x)


    def _get_fourier_series_from_vector(self, x):
        series = {}
        for freq in range(self.no_basis_functions):
            if not isclose(x[freq], 0, abs_tol=TOLERANCE_ZERO):
                series[tuple(self.freqlist[freq])] = x[freq]
        return series

    def get_number_of_measurements(self):
        return self.m

if __name__ == "__main__":
    n = 10
    k = 5
    d = 5
    f = RandomFunction(n, k , d)
    print(f)
    proximal_method = ProximalMethod(n, k, d)
    print(proximal_method.run(f))
