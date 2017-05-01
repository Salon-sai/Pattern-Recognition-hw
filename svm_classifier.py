import data_process
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score
from sklearn.externals import joblib
from os import path
import matplotlib .pyplot as plt

LSVM_PATH = "save_models/lsvm/"

def split_data_set(all_train_data, all_train_label):
    sfk = StratifiedKFold(10)
    all_train_data_array = all_train_data.getA()
    all_train_label_array = all_train_label.ravel()
    for train_index, test_index in sfk.split(all_train_data, all_train_label.ravel()):
        train_data = []; train_label = []
        test_data = []; test_label = []
        for k in train_index:
            train_data.append(all_train_data_array[k])
            train_label.append(all_train_label_array[k])

        for k in test_index:
            test_data.append(all_train_data_array[k])
            test_label.append(all_train_label_array[k])

        yield train_data, train_label, test_data, test_label

def predict_test_accuracy(c_args, all_train_data, all_train_label):
    all_test_data, all_test_label = data_process.all_test_dataAndlabel()
    test_accuracy = []  # 所有C参数对于测试集合的准确率
    precisions = []  # 所有C参数对于测试集合的precision
    recalls = []  # 所有C参数对于测试集合的recall
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
        print("The value of C %1.5f, F1 score : %1.4f" % (c, f1_score(all_test_label, predict_scores)))
    return test_accuracy, precisions, recalls

def cross_validation_c(c_args):
    score_means = []  # 各个C参数的平均准确率
    for c in c_args:
        i = 0
        scores = []  # 当前C参数在不同训练集和测试集中的准确率
        for train_data, train_label, test_data, test_label in split_data_set(all_train_data, all_train_label):
            i += 1
            model_file = LSVM_PATH + "lsvm_" + str(c) + "_" + str(i) + ".pkl"
            if not path.isfile(model_file):
                lsvm = LinearSVC(C=c)
                lsvm.fit(train_data, train_label)
                joblib.dump(lsvm, model_file)
            else:
                lsvm = joblib.load(model_file)

            score = lsvm.score(test_data, test_label)
            scores.append(score)
            # print(model_file, "accuracy_score:", score)
        scores = np.array(scores)
        mean = scores.mean()
        score_means.append(mean)
        # print("\nthe parameters of", c, "mean :", mean, "\n")
    return score_means

if __name__ == '__main__':
    all_train_data, all_train_label = data_process.all_train_dataAndlabel()
    # all_train_data1 = np.concatenate((all_train_data[:100], all_train_data[-100:]))
    # all_train_label1 = np.concatenate((all_train_label[:100], all_train_label[-100:]))

    c_args = [10.0 ** n for n in range(-5, 4)]
    score_means = cross_validation_c(c_args)
    test_accuracy, precisions, recalls = predict_test_accuracy(c_args, all_train_data, all_train_label)

    plt.plot(c_args, score_means, color='blue')
    for i in range(1, len(c_args)):
        c = c_args[i]
        mean = score_means[i]
        plt.scatter(c, mean, 20, color='blue')
        plt.annotate("c: " + str(c) + ", mean: " + str(mean),
                     xy=(c, mean), xycoords='data',
                     xytext=(+10, +30), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel("value of C for SVC")
    plt.ylabel("Cross validated accuracy")
    plt.show()

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

    for i in range(1, len(c_args)):
        plt.plot(recalls[i], precisions[i], color=np.random.rand(3, 1),
                 lw=2.5, label='C value {0}'.format(c_args[i]),)
    plt.plot([0, 1], [0, 1], color='red', lineStyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.show()