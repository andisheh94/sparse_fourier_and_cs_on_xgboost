import numpy as np
from math import ceil, log, isclose, floor
from . utils import naive_cs, efficient_hashing_based_cs, binary_search_cs, reed_solomon_cs, hashing
from . utils.WHT import WHT
from . utils.random_function import RandomFunction
from . utils.cosamp import WHTAlgorithm

class SWHTRobust(object):
    def __init__(self, n, K, C=4, ratio=1.4, finite_field_class="naive_cs", WHT_algorithm = "normal", robust_iterations=3, epsilon = 0.00001,**kwargs):
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
        self.finite_field_cs = None
        # Settings for the finite field decoding
        self.settings_finite_field = kwargs
        # Setting for Walsh-Hadamard transform can be the normal one or cosamp or reduced mfr
        self.WHT_algorithm = WHT_algorithm
        # Number of repetition required for robust iterations
        self.robust_iterations = robust_iterations
        self.eps = epsilon
    def run(self, x):
        # B = no of bins we are hashing to
        # B = 48 * self.K
        B = int(self.K * self.C)
        b = int(ceil(log(B, 2)))
        # T = int(min(floor(log(B, 2)) - 1, ceil(log(self.K, 2)) + 1))
        # T = ceil(log(self.K,4))
        # T = int(min(floor(log(B, 1.6)) - 1, 10*ceil(log(self.K, 2)) + 1))
        #

        # T = int(floor(log(B, self.ratio))) - 1
        T = min(int(floor(log(B, self.ratio))) - 1,7)
        # T = 3
        # current_estimate will hold as key frequencies and as value amplitudes
        current_estimate = {}
        for i in range(T):
            print("B=", B, "b=", b)
            # Define a new hashing matrix A
            if self.finite_field_class == "efficient_hashing_based_cs":
                hash = hashing.InvertibleHashing(self.n, b)
            else:
                hash = hashing.Hashing(self.n, b)
            # hashed_estimate will hold as keys bin frequencies and as values
            # tuples where the first element is the freq hashed to that bin
            # and second element is the amplitude
            hashedEst = self.hash_frequencies(hash, current_estimate)
            residual_estimate = self.detect_frequency(x, hash, hashedEst)

            #########################
            # x.statistic(detectedFreq)
            # bucketCollision = {}
            # for edge in x.graph:
            #     freq = np.zeros((self.n))
            #     freq[edge[0]] = 1
            #     freq[edge[1]] = 1
            #     freq = tuple(freq)
            #     print(edge, "hashed to ", hash.do_FreqHash(freq))
            #     try:
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            #     except KeyError:
            #         bucketCollision[hash.do_FreqHash(freq)] = []
            #         bucketCollision[hash.do_FreqHash(freq)].append(edge)
            # collisions = 0
            # for bucket in bucketCollision:
            #     if len(bucketCollision[bucket]) > 1:
            #         collisions += len(bucketCollision[bucket])
            #         print(bucketCollision[bucket])
            # print("collisions=", collisions)
            ##########################
            # Run iterative updates
            for freq in residual_estimate:
                if freq in current_estimate:
                    current_estimate[freq] = current_estimate[freq] + residual_estimate[freq]
                    if isclose(current_estimate[freq], 0.0, abs_tol=self.eps):
                        # print("deleting", freq)
                        del current_estimate[freq]

                else:
                    current_estimate[freq] = residual_estimate[freq]

            # Buckets sizes for hashing reduces by half for next iteration
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
                #  Initialize empty list
                hashed_estimate[hashed_key] = []
            hashed_estimate[hashed_key].append((key, est[key]))
        return hashed_estimate

    def __get_finite_field_recovery(self, hash):
        # Kaiko hack
        if self.finite_field_cs is not None and self.finite_field_class != "binary_search_cs":
            return self.finite_field_cs

        if self.finite_field_class == "naive_cs":
            return naive_cs.NaiveCS(self.n)
        elif self.finite_field_class == "efficient_hashing_based_cs":
            return efficient_hashing_based_cs.EfficientHashingBasedCS(self.n, hash)
        elif self.finite_field_class == "reed_solomon_cs":
            try:
                return reed_solomon_cs.ReedSolomonCS(self.n, self.settings_finite_field["degree"])
            except KeyError:
                print("For using reed_solomon decoding you need to specify the degree")
        elif self.finite_field_class == "binary_search_cs":
            print("################")
            return binary_search_cs.BinarySearchCS(self.n, **self.settings_finite_field)
        else:
            raise ValueError("The finite_field_class \"", self.finite_field_class, "\"does not exist")
    def get_WHT(self, x, b):
        if self.WHT_algorithm == "normal":
            return WHT(x)
        elif self.WHT_algorithm == "cosamp":
            print("printing x and b",x,b)
            WHT_algorithm = WHTAlgorithm(b, k = (2**b)/self.C ,C=1, algorithm="cosamp")
            out = WHT_algorithm.run(x)
            print("out=", out)
            return out

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
            # print("randomshift", random_shift)
            hashed_signal = hash.do_TimeHash(x, random_shift)
            # print("After Zero shift ", str(x.sampCplx))
            # print("hashed_signal=", hashedSignal)
            ref_signal = self.get_WHT(hashed_signal, hash.b)
            # print(ref_signal)
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
                # print("Bucket", bucket)
                # Compute the values of the current estimation of signal hashed to this bucket and subtract it off the
                # reference signal
                if bucket in hashedEst:
                    for X in hashedEst[bucket]:
                        if self.__inp(X[0], random_shift) == 0:
                            ref_signal[bucket] = ref_signal[bucket] - X[1]
                        else:
                            ref_signal[bucket] = ref_signal[bucket] + X[1]

                        # Only continue if a frequency with non-zero amplitude is hashed to bucket j
                        # print("cheching ref_signal", ref_signal[bucket])
                        # print("ref_signal after subtraction", ref_signal)
                if isclose(ref_signal[bucket], 0.0, abs_tol=self.eps):
                    # print("Entered if statement for ref_signal[bucket]=0")
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

        new_signal_estimate = {}
        # Take majority vote for frequency and median for amplitudes
        # print(detected_frequencies, "\n\n", detected_amplitudes, "\n\n")
        for bucket in measurement_dict:
            if successful_tries[bucket] == 0:
                continue
            # print(successful_tries[bucket])
            measurement = [0] * no_measurements
            for j in range(no_measurements):
                if measurement_dict[bucket][j] > successful_tries[bucket] / 2:
                    measurement[j] = 1
                else:
                    # measurement[j] = 0
                    pass
            try:
                recovered_freq =  finite_field_cs.recover_vector(measurement, bucket)
            except: #Reed solomon degree might be too high
                continue
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
    n = 10
    k = 5
    degree = 2
    # swht = SWHT(n, k)
    swht = SWHTRobust(n, k, finite_field_class="reed_solomon_cs", degree=degree)
    # swht = SWHT(n, k, 1.4, 1.4, finite_field_class="binary_search_cs", no_bins=10, iterations=2)
    # swht = SWHT(n, k, 1.4, 1.4, finite_field_class="hashing_based_cs")
    print("Hellloooo")
    f = RandomFunction(n, k, degree)
    print("f is :", f, flush=True)
    out = swht.run(f)
    print("out is", out)
    fprime = RandomFunction.create_from_FT(n, out)  #  print("fprime is ", fprime)
    if (f == fprime):
        print("Success")
    print(f.get_sampling_complexity())
    # g = Graph(swht.n, swht.K)
    # print(g)
    # p = 0
    # for i in range(1):
    #     # print(i)
    #     y = swht.run(g)
    #     try:
    #         g2 = Graph.create_from_FT(swht.n, y)
    #     except AssertionError:
    #         continue
    #     if g == g2:
    #         p = p+1
    #     g.cache = {}
    # print(p)

    # y = swht.run(g)
    # print(y, "\n", g)
    # y.pop(tuple([0]*swht.n))
    # g2 = Graph.create_from_FT(swht.n, y)
    # print(g == g2)
    # print("SamplingComplexity =", g.sampCplx)
    # bit = 2
    # j = np.arange(20)
    # j = np.floor(j/(2 ** bit))
    # a = (1 - (-1) ** j)/2
    # a = a.astype(int)
    # print(a)
    # print(a.shape)
    # mask = np.zeros(20, dtype=int)
    # mask[4:6] = 1
    # mask[0:2] = 1
    # mask[14:16] = 1
    # print(mask.shape)
    # print(mask)
    # r = np.multiply(a, mask).reshape(20, 1)
    # print(r, r.shape)
    # bitIndexRange = list(range((int(ceil(log(100, 2))))))
    # bitIndexRange.append("ref")
    # print(bitIndexRange)
    # for j in bitIndexRange:
    #     print(j)
