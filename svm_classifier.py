import data_process
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(C=0.5)
clf.fit(data_process.all_train_data, data_process.all_train_label)

predict_labels = clf.predict(data_process.all_test_data)

print(accuracy_score(data_process.all_test_label, predict_labels))