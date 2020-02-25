import pandas as pd
import numpy as np

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.naive_bayes import BernoulliNB
import alsDataManager


# Read the input files and read every line
def loadData(trainingFile, testingFile):
    def convertDataframe(inputFile):
        data = pd.DataFrame(columns=range(100000))

        for i in range(len(inputFile)):
            record = np.fromstring(inputFile[i], dtype=int, sep=' ')
            record_bool = [0 for j in range(100000)]
            for col in record:
                record_bool[col - 1] = 1

            data.loc[i] = record_bool

        return data

    with open(trainingFile, "r") as fr1:
        trainFile = fr1.readlines()

    # Split each line in the two files into label and data
    train_data_list = []
    train_labels_list = []

    for inputData in trainFile:
        train_labels_list.append(inputData[0])

        # Remove the activity label (0/1) and new line character from each record
        inputData = inputData.replace("0\t", "")
        inputData = inputData.replace("1\t", "")
        inputData = inputData.replace("\n", "")
        train_data_list.append(inputData)

    train_labels = np.asarray(train_labels_list)
    train_data = convertDataframe(train_data_list)

    with open(testingFile, "r") as fr2:
        testFile = fr2.readlines()

    test_data = convertDataframe(testFile)

    return train_data, test_data, train_labels


# Project data on a reduced dimensionality k using PCA
def pca(train_data, test_data, k):
    pca = sklearnPCA(n_components=k)
    PCA_projected_trainData = pca.fit_transform(train_data)
    PCA_projected_testData = pca.transform(test_data)

    return PCA_projected_trainData, PCA_projected_testData


# Perform Bernoulli's Naive Bayes Classification
def classifier(PCA_projected_trainData, PCA_projected_testData, train_labels):
    BNBC = BernoulliNB()
    BNBC.fit(PCA_projected_trainData, train_labels)

    predictions = []

    predictions = BNBC.predict(PCA_projected_testData)

    return predictions

#Read the training and the test data set and get 3 separate dataframes of training reviews, test reviews and training labels

dir_path = 'input/dorothea/'

train_data, test_data, train_labels = loadData(dir_path+'train.data', dir_path+'test.data')

output_dict = {}
output_dict['train_data'] = train_data
output_dict['test_data'] = test_data
output_dict['train_labels'] = train_labels
alsDataManager.save_dict_as_json(output_dict = output_dict, output_path = 'dorothea_json.txt')