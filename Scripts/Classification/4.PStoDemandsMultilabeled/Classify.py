import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def Classify(name):
    dataset = pickle.load(open(datasetsDir + name, 'rb'))

    X_train = dataset['train']['vectors'].tolist()
    X_test = dataset['test']['vectors'].tolist()
    Y_train = MultiLabelBinarizer().fit_transform(dataset['train']['classes'])
    Y_test = MultiLabelBinarizer().fit_transform(dataset['test']['classes'])

    ovr = OneVsRestClassifier(GradientBoostingClassifier(ёn_estimators=500,max_depth=4))
    #ovr = OneVsRestClassifier(KNeighborsClassifier())
    #ovr = OneVsRestClassifier(LogisticRegression())
    ovr.fit(X_train, Y_train)
    Y_pred_ovr = ovr.predict(X_test)
    report = classification_report(Y_test, Y_pred_ovr)
    print(name)
    print(report)

datasetsDir ='F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets'
names = ['\\Администратор БД',
         '\\Программист',
         '\\Разработчик Web и мультимедийных приложений',
         '\\Системный администратор ИКТ систем',
         '\\Специалист по большим данным',
         '\\Специалист по интернет-маркетингу',
         '\\Специалист по информационным ресурсам',
         '\\Специалист по тестированию в области информационных технологий']
for name in names:
    filesNames = [name+'.tfidf',
            name+'.d2v',
            name+'.w2v',
            name+'.w2vtfidf',
            name+'.ft'
             ]

    for fname in filesNames:
        Classify(fname)

