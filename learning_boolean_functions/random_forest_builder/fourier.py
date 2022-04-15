from math import isclose
# Numbers smaller than this are considered to be zero and deleted
TOLERANCE_ZERO = 0.00001
# Tolerance used when checking for equality of two Fourier objects
TOLERANCE_EQUALITY = 0.1
from functools import lru_cache
class Fourier:

    def __init__(self, series):
        self.series = series
        self.cleanup()
        self.sampling_complexity = 0

    # Remove frequencies with zero coefficients
    def cleanup(self):
        for key in list(self.series.keys()):
            if isclose(abs(self.series[key]), 0, abs_tol=TOLERANCE_ZERO):
                self.series.pop(key)

    def __str__(self):
        return str(self.series)

    # Creates a Fourier object where keys are tuples not sets
    @classmethod
    def from_tuple_series(cls, series):
        series_with_sets = {}
        for freq, amplitude in series.items():
            new_freq = frozenset([i for i in range(len(freq)) if freq[i]==1])
            series_with_sets[new_freq] = amplitude
        return cls(series_with_sets)

    @classmethod
    def zero(cls):
        return cls({})

    def __getitem__(self, argument):
        self.sampling_complexity += 1
        result = 0
        for key in self.series:
            mult = 1
            for x in key:
                if argument[x] == 1:
                    mult = -mult
            result += mult * self.series[key]
        return result

    def __call__(self, argument):
        return self.__getitem__(argument)

    def get_sampling_complexity(self):
        return self.sampling_complexity

    def reset_sampling_complexity(self):
        self.sampling_complexity = 0

    def degree(self):
        deg = 0
        for key in self.series:
            if len(key) > deg:
                deg = len(key)
        return deg

    def __sub__(self, other):
        new_series = self.series.copy()
        for freq in other.series:
            new_series[freq] = new_series.get(freq, 0) - other.series[freq]
        return Fourier(new_series)

    def __add__(self, other):
        new_series = self.series.copy()
        for freq in other.series:
            new_series[freq] = new_series.get(freq, 0) + other.series[freq]
        return Fourier(new_series)

    def __eq__(self, other):
        for freq in set(self.series.keys()).union(other.series.keys()):
            if not isclose(self.series.get(freq, 0), other.series.get(freq, 0), rel_tol=TOLERANCE_EQUALITY):
                return False
        return True

    def __truediv__(self, divisor):
        return Fourier({freq : amp/divisor for (freq, amp) in self.series.items()})

    def get_sparsity(self):
        return len(self.series)

    @staticmethod
    def get_mse(fourier_1, fourier_2):
        mse = 0
        for freq in set(fourier_1.series.keys()).union(fourier_2.series.keys()):
            mse += (fourier_1.series.get(freq, 0)- fourier_2.series.get(freq, 0)) ** 2

        return mse

    def norm_squared(self):
        return sum([self.series[freq]**2 for freq in self.series.keys()])