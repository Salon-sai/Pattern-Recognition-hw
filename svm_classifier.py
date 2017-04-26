import data_process
from sklearn import svm
from sklearn import cross_validation
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

    print()

    end_time = datetime.datetime.now()
    spend = end_time - start_time

# clf = svm.SVC(C=0.5)
# clf.fit(data_process.all_train_data, data_process.all_train_label)
#
# predict_labels = clf.predict(data_process.all_test_data)
#
# print(accuracy_score(data_process.all_test_label, predict_labels))