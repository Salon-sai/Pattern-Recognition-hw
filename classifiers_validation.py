import data_process
from sklearn import svm
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import datetime

def fit_model(estimators):
    all_train_data, all_train_labels = data_process.all_train_dataAndlabel()
    for k in estimators.keys():
        start_time = datetime.datetime.now()
        estimators[k] = estimators[k].fit(all_train_data, all_train_labels.ravel())
        end_time = datetime.datetime.now()
        spend = end_time - start_time
        print("%s train time: %0.4f" % (k, spend.total_seconds()))

def validation(estimators):
    all_test_data, all_test_labes = data_process.all_test_dataAndlabel()
    for k in estimators.keys():
        start_time = datetime.datetime.now()
        print('-----%s------' % k)
        predict = estimators[k].predict(all_test_data)
        print('%s Score: %0.4f' % (k, estimators[k].score(all_test_data, all_test_labes)))
        print("%s test accuracy %0.4f" % (k, accuracy_score(all_test_labes, predict)))
        end_time = datetime.datetime.now()
        spend = end_time - start_time
        print("%s Time: %0.2f" % (k, spend.total_seconds()))

if __name__ == '__main__':
    estimators = {}
    estimators['bayes'] = GaussianNB()
    estimators['tree'] = tree.DecisionTreeClassifier()
    estimators['forest_100'] = RandomForestClassifier(n_estimators=100)
    estimators['forest_10'] = RandomForestClassifier(n_estimators=10)
    estimators['adaBoost'] = AdaBoostClassifier()
    estimators['svm_c_rbf_05'] = svm.SVC(C=0.5)
    estimators['svm_c_linear'] = svm.SVC(kernel='linear')
    estimators['svm_linear'] = svm.LinearSVC()

    fit_model(estimators)
    print("\n")
    validation(estimators)