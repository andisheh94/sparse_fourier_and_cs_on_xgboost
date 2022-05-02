from dataset_management import dataset_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from random_forest_builder.node import Node
from random_forest_builder.fourier import Fourier
import json
import numpy as np

class RandomForestModel:

    def __init__(self, dataset, n, n_estimators, max_depth):

        data = RandomForestModel.get_dataset(dataset)

        # Use only top n features to train
        try:
            if (dataset=="superconduct" and n==324) or (dataset=="crimes" and n==500):
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
                    (data.x, data.y, test_size=0.1, random_state=1)
            else:
                f = open(f"../random_forest_builder/feature_importance_{dataset}.txt", "r", encoding="utf-8")
                feature_importance = np.array(json.load(f))
                top_n_features = np.argsort(feature_importance)[-n:]
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
                    (data.x, data.y, test_size=0.1, random_state=1)
                self.X_train, self.X_test = self.X_train[:, top_n_features], self.X_test[:, top_n_features]
        except IOError:
            raise Exception(f"Feature importance file for dataset {dataset} does not exist."
                  f"\nPlease make it by running the \"compute_feature_importance\" function ")
        n = self.n_var = self.X_train.shape[1]
        self.shape = [2] * n
        self.sampling_complexity = 0
        self.dataset, self.n_estimators, self.max_depth = dataset, n_estimators, max_depth
        # Split into train and test

        self.regr = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=max_depth, random_state=0)
        self.regr.fit(self.X_train, self.y_train)
        print(f"Fitted random forest regression model with {n_estimators} tree(s), max_depth={max_depth}")
        self.score = self.regr.score(self.X_test, self.y_test)
        print(f"Score on test data is {self.score}")
        self.fourier_transform  = Fourier.zero()
        for decision_tree_regressor in self.regr.estimators_:
            tree = Node.build_tree_from_sklearn(decision_tree_regressor)
            self.fourier_transform += tree.get_fourier()/n_estimators
        print(f"Sparsity = {self.get_fourier_transform().get_sparsity()}")
        self.sparsity = self.get_fourier_transform().get_sparsity()
        self.sampling_complexity = 0
        self.use_cache = False
        self.cache = []
        self.cache_read_index = 0


    def clear_cache(self):
        self.cache = []
        self.cache_read_index = 0

    def reset_sampling_complexity(self):
        self.sampling_complexity = 0

    def get_sampling_complexity(self):
        return self.sampling_complexity

    @staticmethod
    def get_dataset(dataset_name):
        if dataset_name == "crimes":
            data = dataset_regression.CrimesDataset()
        elif dataset_name == "superconduct":
            data = dataset_regression.Superconduct()
        else:
            Exception()
        return data

    def __getitem__(self, item):
        self.sampling_complexity += 1
        if self.use_cache:
            value = self.cache[self.cache_read_index]
            self.cache_read_index += 1
            return value
        else:
            self.cache.append(self.regr.predict(np.reshape(item, (1, -1))).item())
            return self.cache[-1]

    def __call__(self, item):
        return self.__getitem__(self, item)

    def get_fourier_transform(self):
        return self.fourier_transform

    @staticmethod
    def compute_feature_importance(dataset, n_estimators, max_depth):
        """ Computes feature importance of a random forest model trained on the full dimensional dataset"""
        """"results are dumped to a json file"""
        if dataset == "crimes":
            data = dataset_regression.CrimesDataset()
            n_estimators
        elif dataset == "superconduct":
            data = dataset_regression.Superconduct()
        else:
            Exception()
        data = RandomForestModel.get_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.1, random_state=0)
        regression_model = RandomForestRegressor(n_estimators, max_depth=max_depth, random_state=0)
        regression_model.fit(X_train, y_train)
        score = regression_model.score(X_test, y_test)
        print(f"Score on test data is {score}")
        with open(f"feature_importance_{dataset}.txt", "w", encoding="utf-8") as f:
            feature_importances = list(regression_model.feature_importances_)
            json.dump(feature_importances, f)

if __name__ == "__main__":
    result = {}
    dataset = "superconduct"
    for depth in range(1, 13):
        random_forest_model = RandomForestModel(dataset, 324, 20, depth)
        result[depth] = [random_forest_model.get_fourier_transform().get_sparsity(), random_forest_model.score]
        print(result[depth])
        # print({freq:amp for freq, amp in random_forest_model.get_fourier_transform().series.items() if abs(amp)<0.00001})
    with open(f"scores_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
    #RandomForestModel.compute_feature_importance("crimes", 100, 10)
