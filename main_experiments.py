import sys
sys.dont_write_bytecode = True
import os
import pandas as pd
import numpy as np
from src.classifiers import *
from src.calc_metrics import Metrics
from sklearn import preprocessing
from src.smote import SMOTE
from src.mahakil import MAHAKIL
from src.adversarial import AdversarialOversampling
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import time
import pickle
from xlwt import *
import warnings
warnings.filterwarnings("ignore")


def load_data(folder_path):
    files = os.listdir(folder_path)
    data_list, label_list = [], []

    for file in files:
        file_path = folder_path + file
        df = pd.read_csv(file_path)
        df_metrics = df.drop(['bug'], axis=1)
        df.loc[df['bug'] >= 1, 'bug'] = 1
        df_label = df['bug']
        data = np.array(df_metrics.values.tolist())
        label = np.array(df_label.values.tolist())
        data_list.append(data)
        label_list.append(label)

    return data_list, label_list



def calc_performance(label_train, label_pred, pred_proba):
    calc_metrics = Metrics(actual=label_train, prediction=label_pred)
    fpr, tpr, _ = roc_curve(label_train, pred_proba, pos_label=1)
    auc_value = auc(fpr, tpr)

    stats = np.array([j.stats() for j in calc_metrics()])
    stats = stats.flatten()
    recall = stats[0]
    prec = stats[3]
    F_measure = stats[5]
    false_alarm = stats[1]
    G_measure = stats[7]
    bal = stats[8]
    return prec, recall, false_alarm, auc_value, F_measure, G_measure, bal


def save_results(final_results, files):
    i = 0
    for f in files:
        res = f.strip('.csv')
        with open('./dump/' + sampling_method + '/' + res + '.pickle', 'wb') as handle:
            pickle.dump(final_results[i], handle)
            i += 1


def read_results(files):
    results = {}
    for f in files:
        results[f.strip('.csv')] = []

    i = 0
    for f in files:
        res = f.strip('.csv')
        with open('./dump/' + sampling_method + '/' + res + '.pickle', 'rb') as handle:
            results[res] = pickle.load(handle)
            i += 1

    return results


def main_experiment(data_list, label_list, n_splits, n_repeats, sampling_methods, clfs):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    numValues = n_splits * n_repeats
    measures = ["Precision", "Recall", "False_alarm", "AUC", "F_measure", "G_measure", "Bal_measure"]

    # final = {'0': {'clf_1': {'mea_1': [], 'mea_2': [], ...}, {'clf_2': {}}, ...}, '1': {},...}
    final = {}
    for i in range(len(data_list)):
        print("*" * 20)
        print("     Dataset  " + str(i))
        print("*" * 20)

        input_dims = data_list[i].shape[1]


        # result = {'clf_1': {'mea_1': [], 'mea_2': [], ...}, {'clf_2': {}}, ...}, '1': {},...}
        #        = {'clf_1': dic_1, 'clf_2': dic_2,...}
        result = {}
        results = {}

        for sampling_method in sampling_methods:
            for clf in clfs:
                dic = {}
                for q in measures:
                    dic[q] = []
                dic["time"] = []
                result[clf.__name__] = dic
            results[sampling_method] = result

        for train_ind, test_ind in rskf.split(data_list[i], label_list[i]):
            data_train, data_test = data_list[i][train_ind], data_list[i][test_ind]
            label_train, label_test = label_list[i][train_ind], label_list[i][test_ind]

            scaler = preprocessing.MinMaxScaler().fit(data_train)
            data_train = scaler.transform(data_train)
            data_test = scaler.transform(data_test)

            for sampling_method in sampling_methods:

                if sampling_method == 'NoSampling':
                    data_train_, label_train_ = data_train, label_train
                    running_time = 0

                if sampling_method == 'RandomOverSampling':
                    start = time.time()
                    data_train_, label_train_ = RandomOverSampler(random_state=0).fit_sample(data_train, label_train)
                    running_time = time.time() - start

                if sampling_method == 'SMOTE':
                    start = time.time()
                    data_train_, label_train_ = SMOTE().fit_sample(data_train, label_train)
                    running_time = time.time() - start

                if sampling_method == 'MAHAKIL':
                    start = time.time()
                    data_train_, label_train_ = MAHAKIL().fit_sample(data_train, label_train)
                    running_time = time.time() - start

                if sampling_method == 'AdversarialOverSampling':
                    start = time.time()
                    data_train_, label_train_ = AdversarialOversampling(
                        input_dims, input_dims // 2,
                        eps=0.1, pfp=0.5).fit_sample(
                        data_train, label_train,
                        acc_threshold=0.8, shuffle=True)
                    running_time = time.time() - start

                for clf in clfs:
                    print("Classifier   " + clf.__name__ + "  ...")
                    pred, pred_proba = clf(data_train_, label_train_, data_test)

                    prec, recall, false_alarm, auc_value, F_measure, G_measure, bal = \
                        calc_performance(label_test, pred, pred_proba)

                    result[clf.__name__]["Precision"].append(prec)
                    result[clf.__name__]["Recall"].append(recall)
                    result[clf.__name__]["False_alarm"].append(false_alarm)
                    result[clf.__name__]["AUC"].append(auc_value)
                    result[clf.__name__]["F_measure"].append(F_measure)
                    result[clf.__name__]["G_measure"].append(G_measure)
                    result[clf.__name__]["Bal_measure"].append(bal)
                    result[clf.__name__]["time"].append(running_time)

            for clf in clfs:
                for measure in measures:
                    result[clf.__name__][measure] = sum(result[clf.__name__][measure])/numValues
                result[clf.__name__]["time"] = sum(result[clf.__name__]["time"])/numValues

            final[i] = result

    return final


if __name__ == '__main__':
    folder_path = './imbalanced_datasets/'
    data_list, label_list = load_data(folder_path)
    files = os.listdir(folder_path)

    n_splits = 3
    n_repeats = 30

    sampling_methods = ['NoSampling', 'RandomOverSampling', 'SMOTE', 'MAHAKIL', 'AdversarialOverSampling']

    clfs = [KNN, NB, LR, SVM, DT, RF]

    for sampling_method in sampling_methods:
        final_results = main_experiment(data_list, label_list, n_splits, n_repeats, sampling_method, clfs)

        save_results(final_results, files)

        results = read_results(files)

        measures = ["Precision", "Recall", "False_alarm", "AUC", "F_measure", "G_measure", "Bal_measure"]

        workbook = Workbook(encoding='utf-8')

        for file in files:
            booksheet = workbook.add_sheet(file.strip('.csv'))
            i = 0
            for clf in clfs:
                j = 0
                for mea in measures:
                    a = round(results[file.strip('.csv')][clf.__name__][mea], 3)
                    booksheet.write(i, j, a)
                    j += 1
                b = round(results[file.strip('.csv')][clf.__name__]["time"], 3)
                booksheet.write(i, len(measures), b)
                i += 1
        workbook.save('./results/results_'+sampling_method+'.xls')

