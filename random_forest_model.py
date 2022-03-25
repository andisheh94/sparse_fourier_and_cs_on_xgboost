import dataset_regression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import Node
class RandomForestModel:

    def __init__(self, dataset, n_estimators, max_depth=6):
        if dataset == "crimes":
            data = dataset_regression.CrimesDataset()
        elif dataset == "superconduct":
            data = dataset_regression.Superconduct()
        else:
            Exception()

        self.n_var = len(data.shape)
        self.shape = data.shape
        self.sampling_complexity = 0
        self.n_estiamtors, self.max_depth = n_estimators, max_depth
        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
            (data.x, data.y, test_size=0.1, random_state=1)

        self.regr = RandomForestRegressor(max_depth=6, n_estimators=self.n_estiamtors, random_state=0)
        self.regr.fit(self.X_train, self.y_train)
        print(f"Fitted random forest regression model with {n_estimators} tree, max_depth={max_depth}")
        score = self.regr.score(self.X_test, self.y_test)
        print(f"Score on test data is {score}")

        for decision_tree in self.regr.estimators_:
              =  Node.build_tree_from_sklearn(decision_tree).get_fourier







    def __getitem__(self, item):
        item = xgboost.DMatrix(np.array(item).reshape((1, len(item))))
        self.sampling_complexity+=1
        return self.model.predict(item)[0]


    def __call__(self, item):
        return self.__getitem__(self, item)



if __name__ == "__main__":
    boost = RandomForestModel("superconduct", 10, 6)
