import numpy as np
import reedsolo as rs
from reedsolo import rs_calc_syndromes

n = 5
k = 3
r = n - k
p = 3

prim = rs.find_prime_polys(c_exp=p, fast_primes=True, single=True)
rs.init_tables(c_exp=p, prim=prim)
generator_polynomial = rs.rs_generator_poly_all(n)[r]

parity_check_matrix = np.zeros(shape=(r, n), dtype=np.uint16)
generator_matrix = np.zeros(shape=(k, n), dtype=np.uint16)

for i in range(k):
    message = [0] * k
    message[i] = 1
    message = bytearray(message).decode("latin-1")
    message_encoded = rs.rs_encode_msg(message, r, gen=generator_polynomial)
    generator_matrix[i, :] = list(message_encoded)
parity_check_matrix[:, 0:k] = ((2 ** p - 1) & (~generator_matrix[:, k:])).transpose()
parity_check_matrix[:, k:] = np.identity(r, dtype=np.uint16)
print(parity_check_matrix)
print(generator_matrix)
msg = bytearray([1, 1, 0, 0, 0])
print(rs_calc_syndromes(msg, r))
