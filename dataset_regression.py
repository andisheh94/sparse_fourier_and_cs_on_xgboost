import pandas as pd
# import torch
# from torch.utils.data import Dataset
import os
import numpy as np


class Superconduct:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/superconduct/train.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=',')
        print("Accessing superconductor dataset")
        print(file.shape)
        self.samples = 21262
        self.shape= [0] * 81
        x = np.array(file.iloc[1:, 0:81].values)
        y = np.array(file.iloc[1:, 81].values)

        #
        # self.y = y
        #self.x =  x
        # self.shape=[0]*81
        # return
        #

        # Binarize continuous features
        binary_dimension = 0
        split_bits = 4
        split = 2 ** split_bits
        quantile_vector = np.linspace(0, 1, split + 1)
        for col in range(len(self.shape)):
            quantiles = np.quantile(x[:, col], quantile_vector)
            for j in range(split):
                index = np.logical_and(x[:, col] >= quantiles[j],
                                       x[:, col] <= quantiles[j + 1])
                x[index, col] = int(j)
            binary_dimension += split_bits
            self.shape[col] = split
        binary_x = np.zeros((self.samples, binary_dimension), dtype=int)
        temp_index = 0
        for col in range(len(self.shape)):
            feature_size = int(np.ceil(np.log2(self.shape[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x_continuous[row][col])
                binary_x[row, temp_index:temp_index + feature_size] = to_binary(int(x[row, col]),
                                                                                           feature_size)
            temp_index += feature_size

        self.x = binary_x
        self.y = (y-np.mean(y))/np.std(y)
        self.shape = [2] * binary_dimension
        print(len(self.shape))

        print(self.y[0:20])

class BoneMarrow:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/bonemarrow/data.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=',')
        print("Accessing bone-marrow dataset")
        print(file.shape)
        self.samples = 186


        index_continuous = [2,22,28,29,30,31,32,33,34]
        index_categorical = [0,1] + [i for i in range(3,22)] + [i for i in range(23,28)]
        x_categorical = np.array(file.iloc[:, index_categorical].values)
        x_continuous = np.array(file.iloc[:, index_continuous].values)
        y = np.array(file.iloc[:, 35].values)


        #
        #self.y = y
        #self.x =  np.concatenate((x_categorical, x_continuous), axis=1)
        #self.shape=[0]*35
        #return
        #


        self.shape_categorical = [0 for _ in range(len(index_categorical))]
        self.shape_continuous = [0 for _ in range(len(index_continuous))]

        # Binarize categorical features
        binary_dimension_categorical = 0
        for col in range(len(self.shape_categorical)):
            dict = {}
            dim = {}
            temp = 0
            for row in range(self.samples):
                try:
                    dict[x_categorical[row, col]].append(row)
                except KeyError:
                    # print("!!!!!!!",x[row,col])
                    dim[x_categorical[row, col]] = temp
                    temp += 1
                    dict[x_categorical[row, col]] = [row]
            self.shape_categorical[col] = temp
            binary_dimension_categorical += int(np.ceil(np.log2(temp)))
            for key in dim:
                new_value = dim[key]
                index = dict[key]
                x_categorical[index, col] = new_value
        binary_x_categorical = np.zeros((self.samples, binary_dimension_categorical), dtype=int)
        temp_index = 0
        for col in range(len(self.shape_categorical)):
            feature_size = int(np.ceil(np.log2(self.shape_categorical[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            for row in range(self.samples):
                binary_x_categorical[row, temp_index:temp_index + feature_size] = to_binary(x_categorical[row, col],
                                                                                            feature_size)
            temp_index += feature_size

        # preprocess by replacing missing values with the median
        for j in range(3,6):
            index_missing = x_continuous[:,j] == '?'
            index = x_continuous[:,j] != '?'
            temp = np.array(x_continuous[index,j], dtype=float)
            median = np.median(temp)
            x_continuous[index_missing,j]= median
        x_continuous = np.array(x_continuous,dtype=float)
        # Binarize continuous features
        binary_dimension_continuous = 0
        split_bits = 4
        split = 2 ** split_bits
        quantile_vector = np.linspace(0, 1, split + 1)
        for col in range(len(self.shape_continuous)):
            quantiles = np.quantile(x_continuous[:, col], quantile_vector)
            for j in range(split):
                index = np.logical_and(x_continuous[:, col] >= quantiles[j],
                                       x_continuous[:, col] <= quantiles[j + 1])
                x_continuous[index, col] = int(j)
            binary_dimension_continuous += split_bits
            self.shape_continuous[col] = split
        binary_x_continuous = np.zeros((self.samples, binary_dimension_continuous), dtype=int)
        temp_index = 0
        for col in range(len(self.shape_continuous)):
            feature_size = int(np.ceil(np.log2(self.shape_continuous[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x_continuous[row][col])
                binary_x_continuous[row, temp_index:temp_index + feature_size] = to_binary(int(x_continuous[row, col]),
                                                                                           feature_size)
            temp_index += feature_size

        self.x = np.concatenate((binary_x_categorical, binary_x_continuous), axis=1)
        self.y = y
        # print(self.x)
        self.shape = tuple([2] * (binary_dimension_categorical + binary_dimension_continuous ))

class GPS:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/UJIndoorLoc/trainingData.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=',')
        print("Accessing GPS dataset")
        self.samples = 19936
        index_continuous  = [i for i in range(0,520)]
        index_categorical = [523, 524, 525, 526, 527]
        y = np.array(file.iloc[1:, 520].values, dtype=int)
        x_categorical = np.array(file.iloc[1:, index_categorical].values, dtype=int)
        x_continuous = np.array(file.iloc[1:, index_continuous].values, dtype=int)

        #
        # self.y = y
        # self.x =  np.concatenate((x_categorical, x_continuous), axis=1)
        # self.shape=[0]*525
        # return
        #
        print(x_categorical.shape, x_categorical[0,:])
        print(y.shape, y)
        self.shape_categorical = [0 for _ in range(5)]
        self.shape_continuous = [0 for _ in range(520)]

        # Binarize categorical features
        binary_dimension_categorical = 0
        for col in range(0, 5):
            dict = {}
            dim = {}
            temp = 0
            for row in range(self.samples):
                try:
                    dict[x_categorical[row, col]].append(row)
                except KeyError:
                    # print("!!!!!!!",x[row,col])
                    dim[x_categorical[row, col]] = temp
                    temp += 1
                    dict[x_categorical[row, col]] = [row]
            self.shape_categorical[col] = temp
            binary_dimension_categorical += int(np.ceil(np.log2(temp)))
            for key in dim:
                new_value = dim[key]
                index = dict[key]
                x_categorical[index, col] = new_value
        binary_x_categorical = np.zeros((self.samples, binary_dimension_categorical), dtype=int)
        temp_index = 0
        for col in range(0, 5):
            feature_size = int(np.ceil(np.log2(self.shape_categorical[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            for row in range(self.samples):
                binary_x_categorical[row, temp_index:temp_index + feature_size] = to_binary(x_categorical[row, col],
                                                                                            feature_size)
            temp_index += feature_size

        # Binarize continuous features
        binary_dimension_continuous = 0
        split_bits = 6
        split = 2 ** split_bits
        quantiles = np.linspace(-105, 1, split)
        quantiles = np.append(quantiles, 101)
        print("quantiles", quantiles)
        for col in range(520):
            for j in range(split):
                index = np.logical_and(x_continuous[:, col] >= quantiles[j],
                                       x_continuous[:, col] <= quantiles[j + 1])
                x_continuous[index, col] = int(j)
            binary_dimension_continuous += split_bits
            self.shape_continuous[col] = split
        binary_x_continuous = np.zeros((self.samples, binary_dimension_continuous + 520), dtype=int)

        temp_index = 0
        for col in range(520):
            feature_size = int(np.ceil(np.log2(self.shape_continuous[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x_continuous[row][col])
                binary_x_continuous[row, temp_index:temp_index + feature_size] = to_binary(x_continuous[row, col],
                                                                                           feature_size)
                if x_continuous[row,col]!=100:
                    binary_x_continuous[row,-col] =1
            temp_index += feature_size

        np.set_printoptions(threshold=np.inf)
        print(binary_x_continuous[0,:])
        self.x = np.concatenate((binary_x_categorical, binary_x_continuous), axis=1)
        self.y = y
        # print(self.x)
        self.shape = tuple([2] * (binary_dimension_categorical + binary_dimension_continuous+520))


class BlogFeedback:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/BlogFeedback/blogData_train.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=',')
        print("Accessing  BlogFeedback")
        print(file.shape)
        self.samples = 52396

        y = np.array(file.iloc[:, 280].values, dtype=int)
        x = np.array(file.iloc[:, 0:280].values, dtype=int)
        print(x.shape, y.shape)

        #
        # self.x = x
        # self.y = y
        # self.shape = [2]*280
        # return


        #
        cont_features = [i for i in range(0, 62)] + [i for i in range(276, 280)]
        binary_features = [i for i in range(62,276)]
        self.shape = [0 for _ in range(280)]

        # Binarize continuous features
        binary_dimension = 0
        split_bits = 5
        split = 2 ** split_bits
        quantile_vector = np.linspace(0, 1, split + 1)
        for col in cont_features:
            quantiles = np.quantile(x[:, col], quantile_vector)
            # print(quantiles)
            for j in range(split):
                index = np.logical_and(x[:, col] >= quantiles[j],
                                       x[:, col] <= quantiles[j + 1])
                x[index, col] = int(j)
            binary_dimension += split_bits
            self.shape[col] = split

        for col in binary_features:
            index = x[:, col] >= 0.5
            x[index, col] = 1
            binary_dimension += 1
            self.shape[col] = 2
        print(x[0:,binary_features])
        binary_x = np.zeros((self.samples, binary_dimension), dtype=int)
        print("binary dimension=", binary_dimension)
        temp_index = 0
        for col in range(len(self.shape)):
            feature_size = int(np.ceil(np.log2(self.shape[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x[row][col])
                binary_x[row, temp_index:temp_index + feature_size] = to_binary(x[row, col],
                                                                                feature_size)
            temp_index += feature_size
        # print(binary_x_continuous[0:20,0:5])
        self.x = binary_x
        y = (y-np.mean(y))/(np.std(y))
        self.y = y
        # print(self.x[0,:])
        self.shape = tuple([2] * binary_dimension)


class CrimesDataset:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/crimes_dataset/communities.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=',')
        print("Accessing crimes dataset")
        self.samples = 1992

        y = np.array(file.iloc[:, 127].values)
        x = np.array(file.iloc[:, 0:127].values)
        # delete this sample which has a missing value for feature 25
        x = np.delete(x, (129), axis=0)
        y = np.delete(y, (129))
        print(x.shape, y.shape)



        index = [i for i in range(5,101)] + [i  for i in range(118,121)] + [125]
        x = x[:, index]
        # This feature is by default a string and needs to become a float
        for j in range(self.samples):
            x[j,25] = float(x[j, 25])
        self.shape = [0 for _ in range(len(index))]

        #
        #self.x = x
        #self.y = y
        #return

        #


        # Binarize continuous features
        binary_dimension = 0
        split_bits = 5
        split = 2 ** split_bits
        quantile_vector = np.linspace(0, 1, split + 1)
        for col in range(len(self.shape)):
            quantiles = np.quantile(x[:, col], quantile_vector)
            # print(quantiles)
            for j in range(split):
                index = np.logical_and(x[:, col] >= quantiles[j],
                                       x[:, col] <= quantiles[j + 1])
                x[index, col] = int(j)
            binary_dimension += split_bits
            self.shape[col] = split

        binary_x = np.zeros((self.samples, binary_dimension), dtype=int)
        print("binary dimension=", binary_dimension)
        temp_index = 0
        for col in range(len(self.shape)):
            feature_size = int(np.ceil(np.log2(self.shape[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x[row][col])
                binary_x[row, temp_index:temp_index + feature_size] = to_binary(x[row, col],
                                                                                           feature_size)
            temp_index += feature_size
        # print(binary_x_continuous[0:20,0:5])
        self.x = binary_x
        self.y = y
        # print(self.x[0,:])
        self.shape = tuple([2] * binary_dimension)


class DaeguRealEstate:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/Daeg_real_estate/Daegu_Real_Estate_data.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=',')
        # print(file.values[0,:])
        self.samples = 5891

        y = file.iloc[:, 0].values
        x = file.iloc[:,1:].values
        # print(x.shape, y.shape, y)
        x_categorical = x[:, [1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]
        x_continuous = x[:, [0, 3, 4, 8, 9, 13, 14, 26, 27, 28 ]]
        self.shape_categorical = [0 for _ in range(19)]
        self.shape_continuous = [0 for _ in range(10)]

        # Binarize categorical features
        binary_dimension_categorical = 0
        for col in range(0, 10):
            dict = {}
            dim = {}
            temp = 0
            for row in range(self.samples):
                try:
                    dict[x_categorical[row, col]].append(row)
                except KeyError:
                    # print("!!!!!!!",x[row,col])
                    dim[x_categorical[row, col]] = temp
                    temp += 1
                    dict[x_categorical[row, col]] = [row]
            self.shape_categorical[col] = temp
            binary_dimension_categorical += int(np.ceil(np.log2(temp)))
            for key in dim:
                new_value = dim[key]
                index = dict[key]
                x_categorical[index, col] = new_value
        binary_x_categorical = np.zeros((self.samples, binary_dimension_categorical), dtype=int)
        temp_index = 0
        for col in range(0, 10):
            feature_size = int(np.ceil(np.log2(self.shape_categorical[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            for row in range(self.samples):
                binary_x_categorical[row, temp_index:temp_index + feature_size] = to_binary(x_categorical[row, col],
                                                                                            feature_size)
            temp_index += feature_size

        # Binarize continuous features
        binary_dimension_continuous = 0
        split_bits = 5
        split = 2 ** split_bits
        quantile_vector = np.linspace(0, 1, split + 1)
        for col in range(10):
            quantiles = np.quantile(x_continuous[:, col], quantile_vector)
            # print(quantiles)
            for j in range(split):
                index = np.logical_and(x_continuous[:, col] >= quantiles[j],
                                       x_continuous[:, col] <= quantiles[j + 1])
                x_continuous[index, col] = int(j)
            binary_dimension_continuous += split_bits
            self.shape_continuous[col] = split
        binary_x_continuous = np.zeros((self.samples, binary_dimension_continuous), dtype=int)

        temp_index = 0
        for col in range(10):
            feature_size = int(np.ceil(np.log2(self.shape_continuous[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x[row][col])
                binary_x_continuous[row, temp_index:temp_index + feature_size] = to_binary(x_continuous[row, col],
                                                                                           feature_size)
            temp_index += feature_size
        # print(binary_x_continuous[0:20,0:5])
        self.x = np.concatenate((binary_x_categorical, binary_x_continuous), axis=1)
        self.y = y
        # print(self.x)
        self.shape = tuple([2] * (binary_dimension_categorical + binary_dimension_continuous))


class LasvegasHotelDataset:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/LasVegasTripadvisor/LasVegasTripAdvisorReviews-Dataset.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=';')
        self.samples = 504
        y = file.iloc[0:, 4].values
        x = np.concatenate((np.array(file.iloc[0:, :4].values), np.array(file.iloc[0:, 5:].values)), axis = 1)
        print(x.shape)
        x_categorical = x[:,[0,4,5,6,7,8,9,10,11,12,13,15,17,18] ]
        x_continuous = x[:,[1,2,3,14,16]]
        self.shape_categorical = [0 for _ in range(15)]
        self.shape_continuous = [0 for _ in range(5)]

        # Binarize categorical features
        binary_dimension_categorical = 0
        for col in range(0, 14):
            dict = {}
            dim = {}
            temp = 0
            for row in range(self.samples):
                try:
                    dict[x_categorical[row, col]].append(row)
                except KeyError:
                    # print("!!!!!!!",x[row,col])
                    dim[x_categorical[row, col]] = temp
                    temp += 1
                    dict[x_categorical[row, col]] = [row]
            self.shape_categorical[col] = temp
            binary_dimension_categorical += int(np.ceil(np.log2(temp)))
            for key in dim:
                new_value = dim[key]
                index = dict[key]
                x_categorical[index, col] = new_value
        binary_x_categorical = np.zeros((self.samples, binary_dimension_categorical), dtype=int)
        temp_index = 0
        for col in range(0, 14):
            feature_size = int(np.ceil(np.log2(self.shape_categorical[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            for row in range(self.samples):
                binary_x_categorical[row, temp_index:temp_index + feature_size] = to_binary(x_categorical[row, col], feature_size)
            temp_index += feature_size

        # Binarize continuous features
        binary_dimension_continuous = 0
        split_bits = 3
        split = 2 ** split_bits
        quantile_vector = np.linspace(0, 1, split + 1)
        for col in range(5):
            quantiles = np.quantile(x_continuous[:, col], quantile_vector)
            # print(quantiles)
            for j in range(split):
                index = np.logical_and(x_continuous[:, col] >= quantiles[j], x_continuous[:, col] <= quantiles[j + 1])
                x_continuous[index, col] = int(j)
            binary_dimension_continuous += split_bits
            self.shape_continuous[col] = split
        binary_x_continuous = np.zeros((self.samples, binary_dimension_continuous), dtype=int)

        temp_index = 0
        for col in range(0, 5):
            feature_size = int(np.ceil(np.log2(self.shape_continuous[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x[row][col])
                binary_x_continuous[row, temp_index:temp_index + feature_size] = to_binary(x_continuous[row, col], feature_size)
            temp_index += feature_size


        self.x = np.concatenate((binary_x_categorical, binary_x_continuous), axis=1)
        self.y = y
        self.shape = tuple([2] * (binary_dimension_categorical+binary_dimension_continuous))


class GPUDataset:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/SGEMM_GPU/sgemm_product.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=',')
        print(file.shape)
        self.samples = 241599
        self.shape = [0] * 14
        y = file.iloc[1:241600, 14:18].values
        x = file.iloc[1:241600, 0:14].values
        y = np.array(y)
        y = np.median(y, axis=1)
        x = np.array(x, dtype = int)

        binary_dimension = 0
        for col in range(0, 14):
            dict = {}
            dim = {}
            temp = 0
            for row in range(self.samples):
                try:
                    dict[x[row, col]].append(row)
                except KeyError:
                    # print("!!!!!!!",x[row,col])
                    dim[x[row, col]] = temp
                    temp += 1
                    dict[x[row, col]] = [row]
            self.shape[col] = temp
            binary_dimension += int(np.ceil(np.log2(temp)))
            for key in dim:
                new_value = dim[key]
                index = dict[key]
                x[index, col] = new_value

        binary_x = np.zeros((self.samples, binary_dimension), dtype=int)
        temp_index = 0
        for col in range(0, 14):
            feature_size = int(np.ceil(np.log2(self.shape[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            for row in range(self.samples):
                binary_x[row, temp_index:temp_index + feature_size] = to_binary(x[row, col], feature_size)
            temp_index += feature_size
        self.x = binary_x
        self.y = y
        self.shape = tuple([2] * binary_dimension)

class WineDataset:
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        rel_path = "regression_datasets/wine_quality/winequality-white.csv"
        abs_file_path = os.path.join(script_dir, rel_path)
        file = pd.read_csv(abs_file_path, sep=';')
        print(file.shape)
        self.samples = 4598
        self.shape = [0] * 12
        y = file.iloc[1:4599, 11].values
        x = file.iloc[1:4599, 0:11].values

        # replace char values with numbers
        binary_dimension = 0
        split_bits = 3
        split = 2 ** split_bits
        quantile_vector = np.linspace(0, 1 , split+1)
        for col in range(11):
            quantiles = np.quantile(x[:,col], quantile_vector)
            # print(quantiles)
            for j in range(split):
                index = np.logical_and(x[:,col]>=quantiles[j], x[:,col]<= quantiles[j+1])
                x[index, col] = int(j)
            binary_dimension += split_bits
            self.shape[col] = split

        x = np.array(x, dtype=int)
        y = np.array(y)
        # Binarize x
        binary_x = np.zeros((self.samples, binary_dimension), dtype=int)
        temp_index = 0
        for col in range(0, 11):
            feature_size = int(np.ceil(np.log2(self.shape[col])))
            # print(temp_index, "-", temp_index+feature_size-1)
            # print(feature_size)
            for row in range(self.samples):
                # print(x[row][col])
                binary_x[row, temp_index:temp_index + feature_size] = to_binary(x[row, col], feature_size)
            temp_index += feature_size
        self.x = binary_x
        self.shape = tuple([2] * binary_dimension)
        self.y = y

        # print(self.x.shape, self.y.shape)
        # print(self.x, self.y)

def to_binary(number, digits):
    # print(number, digits)
    res = [int(i) for i in list(format(number, '0'+str(digits)+'b'))]
    # print(number, digits,res)
    return res



if __name__ == "__main__":
    #to_binary(20,12)
    m = Superconduct()
    # print(m.binary_x.shape)
    #m = StudentDatasetGenerator()
