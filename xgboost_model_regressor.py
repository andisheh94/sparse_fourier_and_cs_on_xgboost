import xgboost
import dataset_regression
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
class XGBoostModel:
    def __init__(self, dataset="lasvegas_hotel"):

        if dataset == "wine":
            data  = dataset_regression.WineDataset()

        elif dataset == "gpu":
            data = dataset_regression.GPUDataset()
            num_rounds=10
        elif dataset == "lasvegas_hotel":
            data = dataset_regression.LasvegasHotelDataset()
        elif dataset == "real_estate":
            data = dataset_regression.DaeguRealEstate()
        elif dataset == "crimes":
            data = dataset_regression.CrimesDataset()
            num_rounds = 10
            param = {"max_depth": 6}
        elif dataset == "bone_marrow":
            data = dataset_regression.BoneMarrow()

        elif dataset == "blog_feedback":
            data = dataset_regression.BlogFeedback()

        elif dataset == "superconduct":
            data = dataset_regression.Superconduct()
            num_rounds= 10
            param = {"max_depth": 6}
        else:
            Exception()
        self.n_var = len(data.shape)
        self.shape = data.shape
        self.sampling_complexity = 0
        # print("n_var=", self.n_var, "shape=", self.shape)

        X = data.x
        y = data.y
        # print(data.y[0:100])
        train_x, test_x, train_y, test_y = train_test_split \
            (data.x, data.y, test_size=0.1, random_state=1)

        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        print(test_y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_x, test_x, train_y, test_y
        # return None


        #num_round = 100
        dtrain = xgboost.DMatrix(train_x, label=train_y)
        self.bst = xgboost.train(param, dtrain, num_rounds)
        dtest = xgboost.DMatrix(test_x)
        ypred = self.bst.predict(dtest)
        print(ypred[0:20])
        print(self.y_test[0:20])
        # print(self.y_test, ypred)
        print("R^2 score is:",  1 - np.sum((self.y_test - ypred) ** 2)/np.sum((self.y_test-np.mean(self.y_test) ) **2)  )

        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        # ypred, self.y_test.astype(float))

        # print("r-value is ", r_value)
        print("Mean squared error is:",
              np.sqrt(np.sum((self.y_test - ypred) ** 2) / self.y_test.shape[0]))







    def __getitem__(self, item):
        # print("item=", item)
        item = xgboost.DMatrix(np.array(item).reshape((1, len(item))))
        # print("prediction=", self.bst.predict(item)[0])
        self.sampling_complexity+=1
        return self.bst.predict(item)[0]

if __name__ == "__main__":
    boost = XGBoostModel(dataset="crimes")
