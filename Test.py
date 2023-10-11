from Code import *

def Test_Get_Data():
    DF = Get_Data()
    assert DF.shape != None

def Test_Get_Splits(DF):
    xtrain, xtest, ytrain, ytest = Get_Splits(DF.drop('target', axis=1), DF['target'])
    assert xtrain.shape != None
    assert xtest.shape != None
    assert ytrain.shape != None
    assert ytest.shape != None
    return xtrain, xtest, ytrain, ytest

def Test_Get_Metric_Results(xtrain, xtest, ytrain, ytest):
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(xtrain, ytrain)
    ypred = KNN.predict(xtest)
    Get_Metric_Results(ytest, ypred)
    assert accuracy_score(ytest, ypred)*100 != None
    assert precision_score(ytest, ypred, average='macro')*100 != None
    assert recall_score(ytest, ypred, average='macro')*100 != None
    assert f1_score(ytest, ypred, average='macro')*100 != None