import data_process
from sklearn import svm
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import datetime

esimators = {}
esimators['bayes'] = GaussianNB()
esimators['tree'] = tree.DecisionTreeClassifier()
esimators['forest_100'] = RandomForestClassifier(n_estimators=100)
esimators['forest_10'] = RandomForestClassifier(n_estimators=10)
esimators['adaBoost'] = AdaBoostClassifier()
esimators['svm_c_rbf_05'] = svm.SVC(C=0.5)
esimators['svm_c_linear'] = svm.SVC(kernel='linear')
esimators['svm_linear'] = svm.LinearSVC()

for k in esimators.keys():
    start_time = datetime.datetime.now()
    print('-----%s------', k)
    esimators[k] = esimators[k].fit(data_process.all_train_data, data_process.all_train_label)
    predict = esimators[k].predict(data_process.all_test_data)
    print('%s Score: %0.4f' % (k, esimators[k].score(data_process.all_test_data, data_process.all_test_label)))
    print("%s test accuracy %0.4f" % (k, accuracy_score(data_process.all_test_label, predict)))
    # scores = model_selection.cross_val_score(esimators[k], data_process.all_test_data, data_process.all_test_label, cv=5)
    # print("%s Cross Avg. Score: %0.4f (+/- %0.4f)" %(k, scores.mean(), scores.std() * 2))
    end_time = datetime.datetime.now()
    spend = end_time - start_time
    print("%s Time: %0.2f" % (k, spend.total_seconds()))

# clf = svm.SVC(C=0.5)
# clf.fit(data_process.all_train_data, data_process.all_train_label)
#
# predict_labels = clf.predict(data_process.all_test_data)
#
# print(accuracy_score(data_process.all_test_label, predict_labels))