from math import isclose
from functools import cache
TOLERANCE = 0.00001


class Fourier:

    def __init__(self, series):
        self.series = series
        self.cleanup()

    # Remove frequencies with zero coefficients
    def cleanup(self):
        for key in list(self.series.keys()):
            if isclose(abs(self.series[key]), 0, abs_tol=TOLERANCE):
                self.series.pop(key)

    def __str__(self):
        return str(self.series)


    @classmethod
    def zero(cls):
        return cls({})

    @cache
    def __getitem__(self, argument):
        result = 0
        for key in self.series:
            mult = 1
            for x in key:
                if argument[x] == 1:
                    mult = -mult
            result += mult * self.series[key]
        return result


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

