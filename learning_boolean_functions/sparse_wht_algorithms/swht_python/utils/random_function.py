import math
import numpy as np
# from WHT import WHT
import random
from random_forest_builder.fourier import Fourier
class RandomFunction(object):
    def __init__(self, n, k, degree=None, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.n = n
        self.k = k

        if degree == None:
            self.degree = n
        else:
            self.degree = degree
        self.dict = {}
        self.sampCplx = 0
        self.shape = tuple([2] * n)
        for _ in range(k):
            self.add_random_coeff()
            # print("success")
        self.cache = []
        self.use_cache = False
        self.cache_index = 0
        self.fourier_transform = Fourier.from_tuple_series(self.dict)


    def add_random_coeff(self):
        index = list(range(self.n))
        freq_degree = np.random.randint(self.degree, self.degree + 1)
        sampled_coordinates = random.sample(index, freq_degree)
        freq = [0] * self.n
        for i in sampled_coordinates:
            freq[i] = 1
        freq = tuple(freq)
        # print(freq)
        while freq in self.dict:
            # print(freq)
            freq_degree = np.random.randint(0, self.degree + 1)
            sampled_coordinates = random.sample(index, freq_degree)
            freq = [0] * self.n
            for i in sampled_coordinates:
                freq[i] = 1
            freq = tuple(freq)
            # print(freq)
        self.dict[freq] = np.random.randint(-100, 100)

    def add_coeff(self, freq, x):
        if freq not in self.dict:
            self.k += 1
        self.dict[freq] = x

    def __getitem__(self, t):
        if self.use_cache == True:
            value = self.cache[self.cache_index]
            self.cache_index += 1
            self.sampCplx += 1
            return value
        self.sampCplx += 1
        value = 0
        # print("sampling t=", t)

        for freq in self.dict:
            if np.dot(freq, t) % 2 == 0:
                value += self.dict[freq]
            else:
                value -= self.dict[freq]
        self.cache.append(value)
        return value

    def reset_sampling_complexity(self):
        ret = self.get_sampling_complexity()
        self.sampCplx = 0
        return ret

    def get_sampling_complexity(self):
        return self.sampCplx

    def __str__(self):
        return str(self.dict)

    def __eq__(self, other):
        count = 0
        for freq, amplitude in self.dict.items():
            if not math.isclose(amplitude, other.dict.get(freq,0), abs_tol=0.1):
                return False
        for freq, amplitude in other.dict.items():
            if not math.isclose(amplitude, self.dict.get(freq,0), abs_tol=0.1):
                return False
        return True

    def get_fourier_transform(self):
        return self.fourier_transform




# This class has all the properties of RandomFunction except that it returns
# fourier coefficients instead of actual function value


if __name__ == '__main__':
    print(np.dot((10, 0, 1), (1, 2, 1)))
    a = RandomFunction(5, 2, 2)
    # print(a)
    # a[(0, 0, 0, 0, 0)]

    # g = Graph(25, 5)
    # print(g)
    # cut = np.random.randint(0, high=2, size=25)
    # print(cut)
    # print(g[cut])
    # print()
    #
    # g2 = Graph.create_from_FT(25, g.create_dict())
    # print(g == g2)
