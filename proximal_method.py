import apgpy as apg
import numpy as np
from math import log, isclose, factorial
from random_function import RandomFunction
from itertools import combinations
import random
class ProximalMethod():
    def __init__(self, n, k, degree= 2, C=10):
        self.n = n
        self.k = k
        self.C = C
        self.degree = degree
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



        # if self.degree < n:  # intitialze prob array for the low-degree sampler
        #     self.prob_array = np.zeros(self.degree + 1)
        #     sum = 0
        #     for i in range(self.degree + 1):
        #         self.prob_array[i] = self.__nCr(self.n, i)
        #         sum += self.__nCr(self.n, i)
        #     self.prob_array /= sum


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
        # np.random.seed(1)
        # m = No. of measurements
        self.m = int(self.C * self.k * self.degree * log(self.n))
        print("m=", self.m)
        # print("m=", self.m)
        # self.m = 2 ** n
        self.psi = np.zeros((self.m, self.no_basis_functions))
        self.y = np.zeros(self.m)
        dict = {}  # Avoid redundant t
        for j in range(self.m):
            # Generate a random input
            t = [np.random.randint(0, 2) for _ in range(self.n)]
            while tuple(t) in dict:
                t = [np.random.randint(0, 2) for _ in range(self.n)]
            t = tuple(t)
            dict[t] = 1
            for freq in range(self.no_basis_functions):
                self.psi[j, freq] = (-1) ** np.dot(t, self.freqlist[freq])
            self.y[j] = f[t]
            # Subtract off estimate
        # print("sampling finished")
        # print(self.psi, self.y)

    def run(self, f, lmda=0.01):
        self.sample_function(f)
        self.lmda = lmda
        if isclose(np.linalg.norm(self.y), 0, rel_tol=0.001):
            return {}
        x = apg.solve(self.grad, self.proximal, np.zeros(self.no_basis_functions), quiet=True)
        return self.get_fourier_transform(x)


    def get_fourier_transform(self, x):
        fourier = {}
        for freq in range(self.no_basis_functions):
            if not isclose(x[freq], 0, rel_tol=0.001):
                fourier[tuple(self.freqlist[freq])] = x[freq]
        return fourier


if __name__ == "__main__":
    n = 10
    k = 5
    d = 5
    f = RandomFunction(n, k , d)
    print(f)
    proximal_method = ProximalMethod(n, k, d)
    print(proximal_method.run(f))
