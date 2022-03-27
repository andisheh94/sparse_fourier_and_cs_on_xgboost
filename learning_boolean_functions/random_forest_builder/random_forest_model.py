from learning_boolean_functions.dataset_management import dataset_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from node import Node
from fourier import Fourier


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
        self.n_estimators, self.max_depth = n_estimators, max_depth
        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split \
            (data.x, data.y, test_size=0.1, random_state=1)

        self.regr = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=max_depth, random_state=0)
        self.regr.fit(self.X_train, self.y_train)
        print(f"Fitted random forest regression model with {n_estimators} tree, max_depth={max_depth}")
        score = self.regr.score(self.X_test, self.y_test)
        print(f"Score on test data is {score}")

        self.fourier_transform  = Fourier.zero()
        for decision_tree_regressor in self.regr.estimators_:
            tree = Node.build_tree_from_sklearn(decision_tree_regressor)
            print(tree, "\n", tree.get_fourier())
        print(len(tree.get_fourier().series), tree.get_depth(), tree.get_node_count(), tree.get_leaf_count())

    def __getitem__(self, item):
        self.fourier_transform.__getitem__(item)

    def __call__(self, item):
        return self.__getitem__(self, item)



if __name__ == "__main__":
    boost = RandomForestModel("superconduct", 1, 5)

