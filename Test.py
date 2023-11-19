from Code import *


def Test_Get_Data():
    DF = Get_Data()
    assert DF.shape is not None


def Test_Get_Splits(DF):
    xtrain, xtest, ytrain, ytest = Get_Splits(DF.drop('target', axis=1),
                                              DF['target'])
    assert xtrain.shape is not None
    assert xtest.shape is not None
    assert ytrain.shape is not None
    assert ytest.shape is not None
    return xtrain, xtest, ytrain, ytest


def Test_Get_Metric_Results(xtrain, xtest, ytrain, ytest):
    KNN = KNeighborsClassifier(n_neighbors=7)
    KNN.fit(xtrain, ytrain)
    ypred = KNN.predict(xtest)
    Get_Metric_Results(ytest, ypred)
    assert accuracy_score(ytest, ypred) * 100 is not None
    assert precision_score(ytest, ypred, average='macro') * 100 is not None
    assert recall_score(ytest, ypred, average='macro') * 100 is not None
    assert f1_score(ytest, ypred, average='macro') * 100 is not None
