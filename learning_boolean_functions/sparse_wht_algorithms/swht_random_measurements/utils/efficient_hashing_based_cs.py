from .hashing import InvertibleHashing
import numpy as np


class EfficientHashingBasedCS:

    def __init__(self, n, hash):
        # Code block length
        assert (hash.n == n)
        self.n = n
        self.hash = hash

        # Finite field of 2 ** (self.p)
        self.no_binary_measurements = hash.no_binary_measurements
        self.measurement_matrix = hash.measurement_matrix

    def get_measurement_matrix(self):
        return self.measurement_matrix

    def recover_vector(self, measurement_binary, bucket):
        print(measurement_binary, bucket, len(measurement_binary), self.no_binary_measurements)
        if len(measurement_binary) != self.no_binary_measurements:
            raise ValueError("Bin or measurement does not have the correct dimension")
        whole_measurement = np.concatenate((bucket, measurement_binary))
        return self.hash.inverse_freq_hash(whole_measurement)

if __name__ == "__main__":
    hash = InvertibleHashing(5, 2)
    print(hash.whole_matrix, hash.P, hash.measurement_matrix)
    hashing_based_cs = EfficientHashingBasedCS(5, hash)
    print(hashing_based_cs.get_measurement_matrix())
    print(hash.whole_matrix)
    measurement = np.array([0, 0, 1])
    print(hashing_based_cs.recover_vector((0, 1), measurement), hash.whole_matrix.transpose())
