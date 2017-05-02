import data_process
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score
from sklearn.externals import joblib
from os import path
import plot_data
import matplotlib .pyplot as plt

LSVM_PATH = "save_models/lsvm/"

def split_data_set(all_train_data, all_train_label):
    """
    10-folds 划分训练数据集
    :param all_train_data: 所有的训练数据
    :param all_train_label: 所有的训练标签
    :return:
    """
    sfk = StratifiedKFold(10)
    all_train_data_array = all_train_data.getA()
    all_train_label_array = all_train_label.ravel()
    for train_index, test_index in sfk.split(all_train_data, all_train_label.ravel()):
        yield all_train_data_array[train_index], all_train_label_array[train_index], \
            all_train_data_array[test_index], all_train_label_array[test_index]

def predict_test_accuracy(c_args, all_train_data, all_train_label):
    all_test_data, all_test_label = data_process.all_test_dataAndlabel()
    test_accuracy = []  # 所有C参数对于测试集合的准确率
    precisions = []     # 所有C参数对于测试集合的precision
    recalls = []        # 所有C参数对于测试集合的recall
    for c in c_args:
        model_file = LSVM_PATH + "final/lsvm_" + str(c) + "_final.pkl"
        if not path.isfile(model_file):
            lsvm = LinearSVC(C=c)
            lsvm.fit(all_train_data, all_train_label.ravel())
            joblib.dump(lsvm, model_file)
        else:
            lsvm = joblib.load(model_file)
        predict_scores = lsvm.predict(all_test_data)
        test_accuracy.append(accuracy_score(all_test_label.ravel(), predict_scores))
        precision, recall, _ = precision_recall_curve(all_test_label.ravel(), predict_scores)
        precisions.append(precision)
        recalls.append(recall)
    return test_accuracy, precisions, recalls

def cross_validation_c(c_args):
    """
    通过交叉验证计算 accuracy, precision, recall, f1-measure 四个度量值的均值
    :param c_args: 参数C的组数
    :return: accuracy, precision, recall, f1-measure 四个度量值的均值，其中precision, recall用于绘画P-R图
    """
    accuracy_means = []     # 各个C参数的平均准确率
    precision_means = []    # 各个C参数的平均precision值
    recall_means = []       # 各个C参数的平均recall值
    f1_means = []           # 各个C参数的平均f1值
    for c in c_args:
        i = 0
        accuracies = []     # 当前C参数在不同训练集训练，并通过测试集校验的准确率
        precisions = []     # 当前C参数在不同训练集训练，并通过测试集校验的precision
        recalls = []        # 当前C参数在不同训练集训练，并通过测试集校验的recall
        f1_scores = []       # 当前C参数在不同训练集训练，并通过测试集校验的f1-score
        for train_data, train_label, test_data, test_label in split_data_set(all_train_data, all_train_label):
            i += 1
            model_file = LSVM_PATH + "lsvm_" + str(c) + "_" + str(i) + ".pkl"
            # 训练或获取之前用该训练集训练好的模型
            if not path.isfile(model_file):
                lsvm = LinearSVC(C=c)
                lsvm.fit(train_data, train_label)
                joblib.dump(lsvm, model_file)
            else:
                lsvm = joblib.load(model_file)
            predict = lsvm.predict(test_data)   # 预测测试集

            accuracy = accuracy_score(test_label, predict)
            precision, recall, _ = precision_recall_curve(test_label, predict)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score(test_label, predict))

        # 计算当前C参数的各项度量的平均值，并放入均值数组当中
        precision_means.append(np.array(precisions).mean(axis=0))
        recall_means.append(np.array(recalls).mean(axis=0))
        accuracy_means.append(np.array(accuracies).mean())
        f1_means.append(np.array(f1_scores).mean())
    return accuracy_means, precision_means, recall_means, f1_means

if __name__ == '__main__':
    all_train_data, all_train_label = data_process.all_train_dataAndlabel()  # 获取所有训练数据和相应的标签
    c_args = [10.0 ** n for n in range(-5, 4)]  # 生成C参数的数组

    accuracy_means, precision_means, recall_means, f1_means = cross_validation_c(c_args)
    test_accuracy, _, _ = predict_test_accuracy(c_args, all_train_data, all_train_label)

    for i in range(0, len(c_args)):
        print("c value is : %1.5f , f1 score: %1.5f" % (c_args[i], f1_means[i]))

    #  画出交叉校验的准确率均值
    plt.plot(c_args, accuracy_means, color='blue')
    for i in range(1, len(c_args)):
        c = c_args[i]
        mean = accuracy_means[i]
        plt.scatter(c, mean, 20, color='blue')
        plt.annotate("c: " + str(c) + ", mean: " + str(mean),
                     xy=(c, mean), xycoords='data',
                     xytext=(+10, +30), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel("value of C for SVC")
    plt.ylabel("Cross validated accuracy")
    plt.show()

    #  画出测试集合的准确率均值
    plt.plot(c_args, test_accuracy, color='red')
    for i in range(1, len(c_args)):
        c = c_args[i]
        score = test_accuracy[i]
        plt.scatter(c, score, 20, color='blue')
        plt.annotate("c: " + str(c) + ", mean: " + str(score),
                     xy=(c, score), xycoords='data',
                     xytext=(+10, +30), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel("value of C for SVC")
    plt.ylabel("test data set accuracy")
    plt.show()

    #  通过测试集合，计算precision, recall值并画出PR图
    plot_data.plot_precision_recall(["C value {0}".format(c) for c in c_args], precision_means, recall_means)
    # for i in range(0, len(c_args)):
    #     plt.plot(recall_means[i], precision_means[i], color=np.random.rand(3, 1),
    #              lw=2.5, label='C value {0}'.format(c_args[i]),)
    # plt.plot([0, 1], [0, 1], color='red', lineStyle='--')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.legend(loc='lower right')
    # plt.show()