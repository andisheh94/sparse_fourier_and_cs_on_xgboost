import numpy as np
from math import ceil, log, isclose, floor
from sparse_wht_algorithms.swht_python.utils import naive_cs, binary_search_cs, reed_solomon_cs, random_cs, hashing
from sparse_wht_algorithms.swht_python.utils.WHT import WHT
from sparse_wht_algorithms.swht_python.utils.random_function import RandomFunction
import multiprocessing

class SWHTRobust(object):
    def __init__(self, n, K, C=1.4, ratio=1.4, finite_field_class="naive_cs", robust_iterations=1, epsilon=0.0001,
                 no_processes=1, **kwargs):
        # C bucket constant
        self.C = C
        # Size of Ground set
        self.n = n
        # Sparsity
        self.K = K
        # Bucket halving ratio
        self.ratio = ratio
        # What to use for the finite field decoding
        self.finite_field_class = finite_field_class
        # Corresponding Class
        self.finite_field_cs = None
        # Settings for the finite field decoding
        self.settings_finite_field = kwargs
        # Number of repetition required for robust iterations
        self.robust_iterations = robust_iterations
        self.eps = epsilon
        # Parallelized frequency recovery if no_processes is > 1
        self.no_processes = no_processes

    def run(self, x, seed=None):
        if seed!=None:
            np.random.seed(seed)
        # B = no of bins we are hashing to
        B = int(self.K * self.C)
        b = int(ceil(log(B, 2)))
        # No. rounds
        T = min(int(floor(log(B, self.ratio))) - 1,3)
        current_estimate = {}
        for i in range(T):
            # Define a new hashing matrix
            hash = hashing.Hashing(self.n, b)
            # hashed_estimate will hold as keys bin frequencies and as values
            # tuples where the first element is the freq hashed to that bin
            # and second element is the amplitude
            hashedEst = self.hash_frequencies(hash, current_estimate)
            residual_estimate = self.detect_frequency(x, hash, hashedEst)

            # Run iterative updates
            for freq in residual_estimate:
                if freq in current_estimate:
                    current_estimate[freq] = current_estimate[freq] + residual_estimate[freq]
                    if isclose(current_estimate[freq], 0.0, abs_tol=self.eps):
                        del current_estimate[freq]

                else:
                    current_estimate[freq] = residual_estimate[freq]

            # Bucket sizes for hashing reduces by half for next iteration
            B = int(ceil(B / self.ratio))
            b = int(ceil(log(B, 2)))
        return current_estimate

    def hash_frequencies(self, hash, est):
        # This function hashes the current estimated frequencies
        # of the signal to the buckets
        hashed_estimate = {}
        for key in est:
            hashed_key = hash.do_FreqHash(key)
            if hashed_key not in hashed_estimate:
                hashed_estimate[hashed_key] = []
            hashed_estimate[hashed_key].append((key, est[key]))
        return hashed_estimate

    def __get_finite_field_recovery(self, hash):
        # Cached
        if self.finite_field_cs is not None:
            return self.finite_field_cs

        if self.finite_field_class == "naive_cs":
            self.finite_field_cs = naive_cs.NaiveCS(self.n)
            return self.finite_field_cs
        elif self.finite_field_class == "reed_solomon_cs":
            try:
                self.finite_field_cs = reed_solomon_cs.ReedSolomonCS(self.n, self.settings_finite_field["degree"])
                return self.finite_field_cs
            except KeyError:
                print("For using reed_solomon decoding you need to specify the degree")
        elif self.finite_field_class == "binary_search_cs":
            return binary_search_cs.BinarySearchCS(self.n, **self.settings_finite_field)
        elif self.finite_field_class == "random_cs":
            try:
                return random_cs.RandomCS(self.n, hash, self.settings_finite_field["degree"],
                                      self.settings_finite_field["sampling_factor"])
            except KeyError:
                print("For using random_cs decoding you need to specify the degree, sampling factor and wait time")
        else:
            raise ValueError("The finite_field_class \"", self.finite_field_class, "\"does not exist")

    def get_WHT(self, x, b):
        return WHT(x)

    def detect_frequency(self, x, hash, hashedEst):
        # Finite field CS measurement list
        finite_field_cs = self.__get_finite_field_recovery(hash)
        # Subsample Signal
        no_measurements = finite_field_cs.no_binary_measurements
        measurement_dict = {}
        ampl_dict = {}
        successful_tries = {}
        successful_try_random_shift = {}
        for i in range(hash.B):
            bucket = self.toBinaryTuple(i, hash.b)
            # the measurements made in this bin
            measurement_dict[bucket] = np.array([0] * no_measurements, dtype=int)
            # list of amplitudes in this bin each corresponding to a different random shift
            ampl_dict[bucket] = []
            # The number of times the sum of frequencies mapped to the bin (with the random shifts) exceeded the epsilon threshold
            successful_tries[bucket] = 0
            # the corresponding shift of the successful try
            successful_try_random_shift[bucket] = []
        random_shift_list = [np.random.randint(low=0, high=2, size=(self.n,)) for _ in range(self.robust_iterations)]

        for random_shift in random_shift_list:
            hashed_signal = hash.do_TimeHash(x, random_shift)
            ref_signal = self.get_WHT(hashed_signal, hash.b)
            # This dictionary will hold the WHTs of the subsampled signals
            hashedWHT = {}
            # Subsample Signal
            for j in range(no_measurements):
                a = finite_field_cs.get_measurement_matrix()[j, :]
                # print("Measurement=", a)
                hashedSignal = hash.do_TimeHash(x, (a + random_shift)%2)
                hashedWHT[j] = self.get_WHT(hashedSignal, hash.b)
                # print("WHT of measurement=", hashedWHT[j])

            # i is the number of the bucket we are checking in the iterations below
            for i in range(hash.B):
                bucket = self.toBinaryTuple(i, hash.b)
                # Compute the values of the current estimation of signal hashed to this bucket and subtract it off the
                # reference signal
                if bucket in hashedEst:
                    for X in hashedEst[bucket]:
                        if self.__inp(X[0], random_shift) == 0:
                            ref_signal[bucket] = ref_signal[bucket] - X[1]
                        else:
                            ref_signal[bucket] = ref_signal[bucket] + X[1]

                # Only continue if a frequency with non-zero amplitude is hashed to bucket j
                if isclose(ref_signal[bucket], 0.0, abs_tol=self.eps):
                    continue
                else:
                    successful_tries[bucket] += 1
                    successful_try_random_shift[bucket].append(random_shift)

                if bucket in hashedEst:
                    for j in range(no_measurements):
                        for X in hashedEst[bucket]:
                            if self.__inp(X[0], finite_field_cs.get_measurement_matrix()[j, :] + random_shift) == 0:
                                hashedWHT[j][bucket] = hashedWHT[j][bucket] - X[1]
                            else:
                                hashedWHT[j][bucket] = hashedWHT[j][bucket] + X[1]
                for j in range(no_measurements):
                    if np.sign(hashedWHT[j][bucket]) != np.sign(ref_signal[bucket]):
                        try:
                            measurement_dict[bucket][j] += 1
                        except KeyError:
                            measurement_dict[bucket][j] = 1
                ampl_dict[bucket].append(ref_signal[bucket])

        # Get measurements in each bucket
        measurement = {}
        for bucket in measurement_dict:
            if successful_tries[bucket] == 0:
                continue
            measurement[bucket] = [0] * no_measurements
            for j in range(no_measurements):
                if measurement_dict[bucket][j] > successful_tries[bucket] / 2:
                    measurement[bucket][j] = 1
                else:
                    # measurement[bucket][j] = 0
                    pass
        # Launch parallel jobs to get frequency in each bucket
        queue_in = multiprocessing.Queue()
        queue_out = multiprocessing.Queue()
        job_list = []
        for bucket in measurement:
            queue_in.put((measurement[bucket], bucket))
            p = multiprocessing.Process(target=finite_field_cs.recover_vector, args=(queue_in, queue_out))
            job_list.append(p)

        # Start jobs in batches of size self.no_processes
        while job_list:
            batch = []
            for _ in range(self.no_processes):
                try:
                    p = job_list.pop()
                    p.start()
                    batch.append(p)
                except IndexError: #job_list is now empty
                    break
            for p in batch:
                if self.finite_field_class == "random_cs":
                    # Wait for 'wait_time' seconds or until process finishes
                    p.join(self.settings_finite_field["wait_time"])
                    # If thread is still active
                    if p.is_alive():
                        print("alive")
                        p.kill()
                        p.join()
                else:
                    p.join()
        # Make new signal estimate by taking medians
        new_signal_estimate = {}
        while queue_out:
            bucket, recovered_freq = queue_out.get()
            if hash.do_FreqHash(recovered_freq) != bucket:
                continue
            index = 0
            for random_shift in successful_try_random_shift[bucket]:
                if self.__inp(recovered_freq, random_shift) == 1:
                    ampl_dict[bucket][index] = -ampl_dict[bucket][index]
                index += 1
            recovered_ampl = np.median(ampl_dict[bucket])
            new_signal_estimate[tuple(recovered_freq)] = recovered_ampl

        return new_signal_estimate


    # This function computes the inner product of two 0-1 n-tuples
    @staticmethod
    def __inp(a, b):
        # print("inp", size(a), size(b))
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) % 2

    def toBinaryTuple(self, i, b):
        # Converts integer i into an b-tuple of 0,1s
        a = list(bin(i)[2:].zfill(b))
        a = tuple([int(x) for x in a])
        return a


if __name__ == "__main__":
    # np.random.seed(0)
    n = 20
    k = 10
    degree = 3
    swht = SWHTRobust(n, k, finite_field_class="random_cs", degree=degree, sampling_factor=1, wait_time=None )
    # swht = SWHTRobust(n, k, finite_field_class="reed_solomon_cs", degree=degree)
    # swht = SWHTRobust(n, k, finite_field_class="binary_search_cs", cs_bins=30, cs_iterations=2, cs_ratio=1.5)
    f = RandomFunction(n, k, degree)
    #print("f is :", f, flush=True)
    out = swht.run(f)
    #print("out is", out)
    fprime = RandomFunction.create_from_FT(n, out)  #  print("fprime is ", fprime)
    if (f == fprime):
        print("Success")
    print(f.get_sampling_complexity())
