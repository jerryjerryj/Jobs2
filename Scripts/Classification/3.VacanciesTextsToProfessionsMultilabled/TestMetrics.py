import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

classes = [[0],[0],[0],[1],[1],[1],[0,1],[0,1],[0,1]]
y_test = MultiLabelBinarizer().fit_transform(classes)

predictions = [[0],[0],[1],[1],[0],[0],[0],[1],[0,1]]
y_pred = MultiLabelBinarizer().fit_transform(predictions)

report = classification_report(y_test, y_pred)
print(report)