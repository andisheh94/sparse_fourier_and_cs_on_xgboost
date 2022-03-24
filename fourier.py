import numpy as np
from math import isclose
# Represents a Fourier series of a function $\mathbb{F}_2^n \rightarrow \mathbb{R}$, where $n = n_var$ (note that below we identify elements of \mathbb{F}_2^n with subsets of {1, 2, ..., n})
# The represented Fourier series is $\sum_{F\in series.keys()} series[F]\cdot \chi_F$,
# where $\chi_F: \mathbb{F}_2^n \rightarrow \mathbb{R}$ is given by $\chi_F(S) = (-1)^{\sum_{i=1}^n F_i\cdot S_i}$
class Fourier:

    def __init__(self, series):
        # Keys are dictonaries where keys are frozen sets containing coordinates of the frequency and values are
        # the fourier coefficients
        self.series = series
        self.cleanup()

    # Remove frequencies with zero coefficients
    def cleanup(self):
        for key in list(self.series.keys()):
            if isclose(abs(self.series[key]), 0, abs_tol=0.00001 ) :
                self.series.pop(key)

    def __str__(self):
        return str(self.series)

        # Evaluate the series on argument

    def predict(self, matrix):
        if len(matrix.shape) ==1:
            matrix = np.reshape(matrix, newshape = (1, matrix.shape[0]))
            # print("neq matrix shape is", matrix.shape)
        y_pred = np.zeros(matrix.shape[0])
        for row in range(matrix.shape[0]):
            y_pred[row] = self.__getitem__(matrix[row,:])
            if row%1000 == 1:
                print(int(row/1000),"k")
        return y_pred


    def __getitem__(self, argument):
        result = 0
        for key in self.series:
            mult = 1
            for x in key:
                if argument[x] == 1:
                    mult = -mult
            result += mult * self.series[key]
        return result



    # Returns the degree of this Fourier series (len of the largest key)
    def degree(self):
        deg = 0
        for key in self.series:
            if len(key) > deg:
                deg = len(key)
        return deg

    # Returns the sum of the squares of the coefficients of this Fourier series not including
    # the zero frequency
    def norm_nonconstant(self):
        result = 0
        for key in self.series:
            if key != set():
                result += self.series[key] ** 2
        return result

    def __sub__(self, other):
        new_series = {}
        for freq in self.series:
            new_series[freq] = self.series[freq] - other.series.get(freq,0)
        for freq in other.series:
            new_series[freq] = self.series.get(freq,0) - other.series[freq]
        return Fourier(new_series)
    def __add__(self, other):
        new_series = {}
        for freq in self.series:
            new_series[freq] = self.series[freq] + other.series.get(freq,0)
        for freq in other.series:
            new_series[freq] = self.series.get(freq,0) + other.series[freq]
        return Fourier(new_series)

    # Returns the sum of the squares of the coefficients of this Fourier series
    def norm(self):
        result = 0
        for key in self.series:
            result += self.series[key] * self.series[key]
        return result
