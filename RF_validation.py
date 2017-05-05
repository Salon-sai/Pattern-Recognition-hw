# -*- coding: UTF-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score
import numpy as np
import data_process
import plot_data
from sklearn.externals import joblib
from os import path
import datetime

MODEL_PATH = "save_models/RandomForest/"

def cross_validation_args(suffix, classifier, all_train_data, all_train_label):
    accuracies = []     # 存放各个训练集合所得出的准确率
    precisions = []     # 存放各个训练集合所得出的precision
    recalls = []        # 存放各个训练集合所得出的recall
    f1_scores = []      # 存放各个训练集合所得到的f1值

    cross_data_label = [x for x in data_process.split_data_set(all_train_data, all_train_label)]
    i = 1
    for train_x, train_y, test_x, test_y in cross_data_label:
        classifier = fitOrload_model(suffix + "_" + str(i), classifier, train_x, train_y)
        predict = classifier.predict(test_x)

        accuracies.append(accuracy_score(test_y, predict))
        precision, recall, _ = precision_recall_curve(test_y, predict)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score(test_y, predict))
        i += 1
    return np.array(accuracies).mean(), np.array(precisions).mean(axis=0),\
           np.array(recalls).mean(axis=0), np.array(f1_scores).mean()

def fitOrload_model(suffix, classifier, train_x, train_y):
    model_file = "%sRandomForest_%s.pkl" % (MODEL_PATH, suffix)
    if path.isfile(model_file):
        classifier = joblib.load(model_file)
    else:
        start_time = datetime.datetime.now()
        classifier.fit(train_x, train_y)
        end_time = datetime.datetime.now()
        print("%s spend time ：%1.3f" % (suffix, (end_time - start_time).seconds))
        joblib.dump(classifier, model_file)
    return classifier


if __name__ == '__main__':
    all_train_data, all_train_label = data_process.all_train_dataAndlabel()
    # 设置森林中树的个数
    n_estimators_list = [20, 50]
    # 设置最大的特征参数为 10, log2(n_features), sqrt(n_features), n_features
    max_features_list = [10, "log2", "sqrt", 100, 200]
    # 存放不同参数的随机森林的分类器
    rf_dict = dict()

    # accuracy_means = dict()
    # precision_means = dict()
    # recall_means = dict()
    # f1_means = dict()

    n_estimators_map = dict()
    for n_estimators in n_estimators_list:
        measure_map = dict()
        n_estimators_map[n_estimators] = measure_map

        for max_features in max_features_list:
            key = "{0}_{1}".format(n_estimators, max_features)
            rf_dict[key] = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
            accuracy_mean, precision_mean, recall_mean, f1_mean = \
                cross_validation_args(key, rf_dict[key], all_train_data, all_train_label)

            measure_map[max_features] = dict()

            measure_map[max_features]["accuracy_means"] = accuracy_mean
            measure_map[max_features]["precision_means"] = precision_mean
            measure_map[max_features]["recall_means"] = recall_mean
            measure_map[max_features]["f1_means"] = f1_mean

            # accuracy_means[key] = accuracy_mean
            # precision_means[key] = precision_mean
            # recall_means[key] = recall_mean
            # f1_means[key] = f1_mean


    # 打印不同参数森林的f1值
    for n_estimators, measure_map in n_estimators_map.items():
        print("=============================")
        print("the number of tree is : %s" % (n_estimators))
        for max_features, measures in measure_map.items():
            print("------------------------------")
            print("the max_feature is : %s" % (max_features))
            print("the accuracy mean : %1.4f, f1 score : %1.4f" % (measures["accuracy_means"], measures["f1_means"]))

        plot_data.plot_precision_recall(title="the number of tree : %s " % n_estimators,
                                        plot_labels=["max feature : %s" % feature_num for feature_num in measure_map.keys()],
                                        precisions=[measures["precision_means"] for measures in measure_map.values()],
                                        recalls=[measures['recall_means'] for measures in measure_map.values()])

    # plot_data.plot_precision_recall(["max feature : %s" % str(x) for x in f1_means.keys()],
    #                                 [precision for precision in precision_means.values()],
    #                                 [recall for recall in recall_means.values()])