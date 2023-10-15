# Importing All Required Libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

# Loading the dataset into a dataframe
# Importing the dataset from sklearn


def Get_Data():

    dataset = load_iris()
    X, Y = dataset.data, dataset.target

    X = pd.DataFrame(X, columns=dataset.feature_names)
    Y = pd.DataFrame(Y, columns=['target'])
    DF = pd.concat([X, Y], axis=1)
    return DF


DF = Get_Data()
print(DF)

# Checking Distribution of Target Variable
DF.target.value_counts()


def Get_Splits(X, Y):
    # Splitting the Dataset
    xtrain, xtest, ytrain, ytest = train_test_split(X,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=0)

    ytrain = ytrain.values.ravel()
    ytest = ytest.values.ravel()
    return xtrain, xtest, ytrain, ytest


xtrain, xtest, ytrain, ytest = Get_Splits(DF.drop('target', axis=1),
                                          DF['target'])
print(np.shape(xtrain))
print(np.shape(xtest))
print(np.shape(ytrain))
print(np.shape(ytest))


# Training and Testing a Classifier
def Get_Metric_Results(ytest, ypred):
    acc = accuracy_score(ytest, ypred) * 100
    prec = precision_score(ytest, ypred, average='macro') * 100
    rec = recall_score(ytest, ypred, average='macro') * 100
    f1 = f1_score(ytest, ypred, average='macro') * 100
    return acc, prec, rec, f1


KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(xtrain, ytrain)
ypred = KNN.predict(xtest)
acc, prec, rec, f1 = Get_Metric_Results(ytest, ypred)
print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)
print("F1 Score: ", f1)
filename = 'Sklearn_Model.pkl'
pickle.dump(KNN, open(filename, 'wb'))
