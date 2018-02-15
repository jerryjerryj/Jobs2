import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def Classify(path, name):
    dataset = pickle.load(open(datasetsDir + name, 'rb'))

    X_train = dataset['train']['vectors'].tolist()
    X_test = dataset['test']['vectors'].tolist()
    Y_train = MultiLabelBinarizer().fit_transform(dataset['train']['classes'])
    Y_test = MultiLabelBinarizer().fit_transform(dataset['test']['classes'])

    ovr = OneVsRestClassifier(LogisticRegression())
    ovr.fit(X_train, Y_train)
    Y_pred_ovr = ovr.predict(X_test)
    report = classification_report(Y_test, Y_pred_ovr)
    print(name)
    print(report)

datasetsDir ='F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets'
names = ['\\Администратор БДW2VTfIdf.dataset',
        '\\Администратор БДW2V.dataset',
        '\\Администратор БДTfIdf.dataset',]

for name in names:
    Classify(datasetsDir,name)

