import sys

sys.dont_write_bytecode = True
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def DT(train_data, train_label, test_data):
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    pred_proba = model.predict_proba(test_data)[:, 1]
    return pred_label, pred_proba


def KNN(train_data, train_label, test_data):
    model = neighbors.KNeighborsClassifier(n_neighbors=8)
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    pred_proba = model.predict_proba(test_data)[:, 1]
    return pred_label, pred_proba


def LR(train_data, train_label, test_data):
    model = LogisticRegression()
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    pred_proba = model.predict_proba(test_data)[:, 1]
    return pred_label, pred_proba


def SVM(train_data, train_label, test_data):
    model = SVC(kernel='linear', probability=True)
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    pred_proba = model.predict_proba(test_data)[:, 1]
    return pred_label, pred_proba


def NB(train_data, train_label, test_data):
    model = GaussianNB()
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    pred_proba = model.predict_proba(test_data)[:, 1]
    return pred_label, pred_proba


def RF(train_data, train_label, test_data):
    model = RandomForestClassifier(criterion='entropy')
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    pred_proba = model.predict_proba(test_data)[:, 1]
    return pred_label, pred_proba


def MLP(train_data, train_label, test_data):
    model = MLPClassifier(hidden_layer_sizes=(10, ), activation='logistic')
    model.fit(train_data, train_label)
    pred_label = model.predict(test_data)
    pred_proba = model.predict_proba(test_data)[:, 1]
    return pred_label, pred_proba



