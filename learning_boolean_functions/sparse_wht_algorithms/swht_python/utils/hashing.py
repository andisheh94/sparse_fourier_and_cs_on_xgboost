import numpy as np
from numpy import matrix
from numpy import linalg
import math


class Hashing(object):
    def __init__(self, n, b):
        # Permutation Matrix is an n * b array
        self.P = np.random.randint(low=0, high=2, size=(n, b))
        ##print("P is", self.P.transpose())
        # n is dimensionality of signal
        self.n = n
        # B = 2^b = no. of buckets we are hashing into
        self.b = b
        self.B = 2 ** self.b
        self.cache = {}
        # print(self.P)
    def do_TimeHash(self, signal, shift):
        key = tuple(np.squeeze(shift).tolist())
        try:
            return self.cache[key]
        except KeyError:
            pass
        out = np.zeros(shape=tuple([2] * self.b), dtype=np.float64)
        for i in range(self.B):
            t = self.__to_binary(i)
            index = (np.dot(self.P, t) + shift) % 2
            out[tuple(np.squeeze(t))] = signal[tuple(np.squeeze(index))]
        self.cache[key] = out
        return out

    def do_FreqHash(self, freq):
        # print(self.P)
        f = np.array(freq, dtype=np.intc)
        hashed_f = np.dot(np.transpose(self.P), f) % 2
        return tuple(hashed_f.tolist())

    def __to_binary(self, i):
        # Converts integer i into an (n,1) 0-1 vector
        a = list(bin(i)[2:].zfill(self.b))
        a = [int(x) for x in a]
        a = np.array(a, dtype=np.intc)
        return a


if __name__ == "__main__":
    #np.random.seed(10)
    x = np.random.randint(0, high=5, size=(2, 2, 2))
    print(x)
    print(x[(0, 0, 0)])
    print(x[(0, 0, 1)])
    print(x[(0, 1, 0)])
    print(x[(0, 1, 1)])
    print(x[(1, 0, 0)])
    print(x[(1, 0, 1)])
    print(x[(1, 1, 0)])
    print(x[(1, 1, 1)])

    h = Hashing(n=3, b=2)
    shift = np.array([[0], [0], [0]], dtype=np.intc)
    # out = h.do_TimeHash(signal = x, shift = shift)
    # print ("Output is", out)
    x = np.random.random(1024 ** 2)
    print(x.shape)
    h = Hashing(n=4, b=2)
    # (h.__to_binary(642))
    print(h.do_FreqHash((0, 1, 0, 0)))
    h = InvertibleHashing(20, 4)
    print(h.extended_P.shape)
    success = 0
    for _ in range(10000):
        measurment = np.random.randint(low=0, high=2, size=20)
        if h.inverse_freq_hash(measurment):
            success += 1
    print(success)
