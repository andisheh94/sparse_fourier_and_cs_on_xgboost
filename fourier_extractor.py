from sample_optimal_sparse_hadamard.sparseWHT_robust_sampleoptimal import SWHTRobust
from fourier import Fourier
import pickle
import numpy as np
import scipy
class FourierExtractor:
    def __init__(self, model, use_buffer =False, dataset="gpu", epsilon = 0.00001, randomseed=0):
        # model needs to overload the __get_item operator as well as
        # have a n_var variable and a shape variable
        if model != None:
            self.n_var = model.n_var
            self.model = model
        self.use_buffer = use_buffer
        self.dataset = dataset
        self.epsilon = epsilon
        self.randomseed = randomseed
    def get_fourier(self, k=100, algorithm="robustWHT"):
        #return a k-sparse approximation of the model
        self.k = k
        if self.use_buffer:
            with open(f'obj/{self.dataset}{self.randomseed}_k={k}.pkl', 'rb') as f:
                series, self.sampling_complexity = pickle.load(f)
                return Fourier(series)
        if algorithm == "proximal":
            proximal_method = ProximalMethod(self.n_var, k, degree =2)
            fourier_transform = proximal_method.run(self.model)
        elif algorithm == "robustWHT":
            print("n_var", self.n_var)
            sparse_wht = SWHTRobust(self.n_var, k, finite_field_class="reed_solomon_cs", degree=6, robust_iterations=1 , epsilon = self.epsilon)
            self.model.sampling_complexity = 0
            fourier_transform = sparse_wht.run(self.model)

        # need to compile the keys into sets for compatibility with Jan's code
        series = {}
        # print("Fourier=", fourier_transform)
        # TODO: Needs update
        for freq, amplitude in fourier_transform.items():
            series[self.__get_set(freq)] = amplitude
            # print(self.__get_set(freq), amplitude)

        self.sampling_complexity = self.model.sampling_complexity
        with open(f'obj/{self.dataset}{self.randomseed}_k={k}.pkl', 'wb') as f:
            pickle.dump([series, self.sampling_complexity] , f)

        return Fourier(series)




    @staticmethod
    def __get_set(freq):
        freq = list(freq)
        # print("freq", freq)
        index = 0
        set = frozenset()
        for i in freq:
            if i>=1:
                set = set.union([index])
            index+=1
        # print("set", set)
        return set


    def test(self):
        self.use_buffer=True
        fourier = self.get_fourier(k=self.k)
        X_test, y_test = self.model.X_test, self.model.y_test
        ypred = np.array(fourier.predict(X_test))
        R2_test = 1 - np.sum((y_test - ypred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        print("R^2 is:", 1 - np.sum((y_test - ypred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            ypred, y_test.astype(float))
        print("r-value is ", r_value)
        MSE_test =  np.sqrt(np.sum((y_test - ypred) ** 2) / y_test.shape[0])
        print("Mean squared error is:",
              np.sqrt(np.sum((y_test - ypred) ** 2) / y_test.shape[0]))


        print("Checking on random inputs")
        random_samples = 1000
        fourier_pred = np.zeros(shape=(random_samples))
        real_pred = np.zeros(shape=(random_samples))
        for i in range(random_samples):
            sample = np.array([np.random.randint(0, 2) for s in range(self.n_var)])
            fourier_pred[i] = np.squeeze(fourier.predict(sample))
            real_pred[i] = self.model[sample]
        R2_func = 1 - np.sum((fourier_pred - real_pred) ** 2) / np.sum((real_pred - np.mean(real_pred)) ** 2)
        print("R^2 is:", 1 - np.sum((fourier_pred - real_pred) ** 2) / np.sum((real_pred - np.mean(real_pred)) ** 2))
        MSE_func = np.sqrt(np.sum((fourier_pred - real_pred) ** 2) / random_samples)
        print("Mean squared error is:",
              np.sqrt(np.sum((fourier_pred - real_pred) ** 2) / random_samples))
        return R2_test, MSE_test, R2_func, MSE_func
if __name__ == "__main__":
    # model = NeuralNetwork()
    # model = SklearnDecisionTree()
    # model = XGBoostModel(dataset="gpu")
    fourier_extractor = FourierExtractor(None, use_buffer=True)
    series = fourier_extractor.get_fourier(800).series
    print({k: v for k, v in sorted(series.items(), key=lambda item: item[1])})
    print(len(series))

