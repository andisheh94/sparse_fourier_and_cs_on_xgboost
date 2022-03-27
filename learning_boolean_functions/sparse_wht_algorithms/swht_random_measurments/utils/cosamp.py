import numpy as np
from .random_function import RandomFunction


class WHTAlgorithm():
    def __init__(self, n, k, C, algorithm="cosamp"):
        self.n = n
        self.k = k
        self.C = C
        self.algorithm = algorithm

    def __to_binary(self, i):
        # Converts integer i into an (n,1) 0-1 vector
        a = list(bin(i)[2:].zfill(self.n))
        a = [int(x) for x in a]
        a = np.array(a, dtype=np.intc)
        return a

    def sample_signal(self, x):
        # np.random.seed(1)
        # m = No. of measurements
        assert (x.shape == tuple([2] * self.n))
        self.m = min(int(self.C * self.k * self.n),2**self.n)
        print("m=", self.m, "n=", self.n, "k=", self.k, "C=", self.C)
        # psi is the observation matrix
        self.psi = np.zeros((self.m, 2 ** self.n))
        # y = observation vector
        self.y = np.zeros(self.m)
        dict = {}
        for j in range(self.m):
            # Generate a random cut
            cut = [np.random.randint(0, 2) for _ in range(self.n)]
            while tuple(cut) in dict:
                cut = [np.random.randint(0, 2) for _ in range(self.n)]
            dict[tuple(cut)] = 1
            # cut = [0] * self.n
            # for i in range(self.n):
            # cut[i] = np.random.randint(0, 2)
            for i in range(2 ** self.n):
                self.psi[j, i] = (-1) ** np.dot(np.array(cut), self.__to_binary(i))
            self.y[j] = x[tuple(cut)]

    def run(self, x):
        self.sample_signal(x)
        if self.algorithm == "cosamp":
            out = self.__cosamp(self.psi, self.y, self.k)
        elif self.algorithm =="mfr":
            out = self.__modified_MFR(self.psi,self.y,self.k)
        return out.reshape(tuple([2] * self.n))

    @staticmethod
    def __cosamp(phi, u, s, epsilon=1e-10, max_iter=1000):
        """
        Return an `s`-sparse approximation of the target signal
        Input:
            - phi, sampling matrix
            - u, noisy sample vector
            - s, sparsity
        """
        s = int(s)
        a = np.zeros(phi.shape[1])
        v = u
        it = 0  # count
        halt = False
        while not halt:
            it += 1
            print("Iteration {}\r".format(it), end="")

            y = np.dot(np.transpose(phi), v)
            omega = np.argsort(y)[-(2 * s):]  # large components
            omega = np.union1d(omega, a.nonzero()[0])  # use set instead?
            phiT = phi[:, omega]
            b = np.zeros(phi.shape[1])
            # Solve Least Square
            b[omega], _, _, _ = np.linalg.lstsq(phiT, u)

            # Get new estimate
            b[np.argsort(b)[:-s]] = 0
            a = b

            # Halt criterion
            v_old = v
            v = u - np.dot(phi, a)

            halt = (np.linalg.norm(v - v_old) < epsilon) or \
                   np.linalg.norm(v) < epsilon or \
                   it > max_iter

        return a

    @staticmethod
    def __modified_MFR(phi, y, k, step_size=1e-3, epsilon=1e-10):
        m, n = phi.shape[0], phi.shape[1]
        x_new = x_old = np.zeros(n)
        gamma_0 = set([])
        while True:
            hat_x = x_new
            hat_x = hat_x + step_size * np.dot(phi.transpose(), y - np.dot(phi, hat_x))
            print("hat_x", hat_x)
            support = np.abs(hat_x).argsort()[-k:]
            print("support", support)
            gamma = set(support)
            print("gamma=", gamma, "gamma_0=", gamma_0)
            if gamma != gamma_0:
                x_old = x_new.copy()
                x_new[support], _, _, _ = np.linalg.lstsq(phi[:, support], y)
                gamma_0 = gamma
            else:
                x_old = x_new.copy()
                x_new = hat_x

            print("x_new=", x_new, "x_old=", x_old)
            if np.linalg.norm(x_new - x_old) < epsilon:
                break

        threshold_indices = np.absolute(x_new) < 0.01
        x_new[threshold_indices] = 0
        return x_new


if __name__ == "__main__":
    n = 10
    k = 3
    f = RandomFunction(n, k)
    print(f)
    proximal = WHTAlgorithm(n, k, 2, algorithm= "cosamp")
    out = proximal.run(f)
    print(out.shape, out[out>0.1])
