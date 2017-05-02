import data_process
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score
from sklearn.externals import joblib
from os import path
import plot_data
import numpy as np

MODEL_PATH = "save_models/"

def cross_validation_models(estimators, all_train_data, all_train_labels):
    precision_means = []    # 各个模型的平均precision值
    recall_means = []       # 各个模型的平均recall值
    f1_means = dict()       # 各个模型的平均f1_score值

    for k in estimators.keys():
        i = 0
        accuracies = []     # 当前模型的交叉验证的准确率数组
        precisions = []     # 当前模型的交叉验证的precision数组
        recalls = []        # 当前模型的交叉验证的recall数组
        f1_scores = []      # 当前模型的交叉验证的F1数组
        for train_data, train_label, test_data, test_label in \
                data_process.split_data_set(all_train_data, all_train_labels):
            i += 1
            model_file = MODEL_PATH + k + "_" + str(i) + ".pkl"
            if path.isfile(model_file):
                estimators[k] = joblib.load(model_file)
            else:
                estimators[k] = estimators[k].fit(train_data, train_label)
                joblib.dump(estimators[k], model_file)
            predict = estimators[k].predict(test_data)  # 测试数据预测值

            precision, recall, _ = precision_recall_curve(test_label, predict)
            f1_scores.append(f1_score(test_label, predict))
            accuracies.append(accuracy_score(test_label, predict))
            precisions.append(precision)
            recalls.append(recall)

        precision_means.append(np.array(precisions).mean(axis=0))
        recall_means.append(np.array(recalls).mean(axis=0))
        f1_means[k] = np.array(f1_scores).mean()

        # 打印交叉验证平均准确率
        print("%s average accuracy: %1.4f" % (k, np.array(accuracies).mean()))

    return precision_means, recall_means, f1_means

def validataion_with_testSet(estimators):
    test_accuracy = []  # 各个模型关于测试集合的准确率
    precisions = []     # 各个模型关于测试集合的precisions
    recalls = []        # 各个模型关于测试集合的recalls
    all_test_data, all_test_labels = data_process.all_test_dataAndlabel()
    for k in estimators.keys():
        predict_scores = estimators[k].predict(all_test_data)
        test_accuracy.append(accuracy_score(all_test_labels.ravel(), predict_scores))
        precision, recall, _ = precision_recall_curve(all_test_labels.ravel(), predict_scores)
        precisions.append(precision)
        recalls.append(recall)
    return test_accuracy, precisions, recalls

if __name__ == '__main__':
    all_train_data, all_train_labels = data_process.all_train_dataAndlabel()
    estimators = dict()
    estimators['bayes/gaussianNB'] = GaussianNB()
    estimators['tree'] = tree.DecisionTreeClassifier()
    # estimators['forest_100'] = RandomForestClassifier(n_estimators=100)
    # estimators['forest_10'] = RandomForestClassifier(n_estimators=10)
    estimators['adaBoost'] = AdaBoostClassifier()
    estimators['lsvm/lsvm_1e-05'] = svm.LinearSVC(C=10**-5)

    precision_means, recall_means, f1_means = cross_validation_models(estimators, all_train_data, all_train_labels)

    for key, f1 in f1_means.items():
        print("The model %s f1 value : %1.5f" % (key, f1))

    plot_data.plot_precision_recall([k for k in estimators.keys()], precision_means, recall_means)

