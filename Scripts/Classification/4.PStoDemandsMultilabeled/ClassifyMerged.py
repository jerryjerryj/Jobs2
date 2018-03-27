import pickle
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def Classify(dataset):
    X_train, X_test, y_train, y_test = train_test_split(dataset['vectors'],dataset['classes'],test_size=0.3)
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = MultiLabelBinarizer().fit_transform(y_train)
    y_test = MultiLabelBinarizer().fit_transform(y_test)

    #ovr = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=500,max_depth=4))
    #ovr = OneVsRestClassifier(KNeighborsClassifier())
    ovr = OneVsRestClassifier(LogisticRegression())
    ovr.fit(X_train, y_train)
    Y_pred_ovr = ovr.predict(X_test)
    report = classification_report(y_test, Y_pred_ovr)
    print(name)
    print(report)

def CrossClassify(dataset):
    classes = MultiLabelBinarizer().fit_transform(dataset['classes'])
    vectors = dataset['vectors'].tolist()

    ovr = OneVsRestClassifier(LogisticRegression())
    #ovr = OneVsRestClassifier(KNeighborsClassifier())
    #ovr = OneVsRestClassifier(GradientBoostingClassifier())#(n_estimators=500, max_depth=4))
    scores = cross_val_score(ovr, vectors, classes, cv=5, scoring='f1_macro')
    #print(scores)
    #print('Average F1-macro: ' + str(scores.mean()) + '\n')
    return str(scores.mean())

datasetsDir ='F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets'
names = ['\\Администратор БД',
         '\\Программист',
         '\\Разработчик Web и мультимедийных приложений',
         '\\Системный администратор ИКТ систем',
         '\\Специалист по большим данным',
         '\\Специалист по интернет-маркетингу',
         '\\Специалист по информационным ресурсам',
         '\\Специалист по тестированию в области информационных технологий']

extentions = [ '.tfidf',
              '.w2vtfidf',]#'.d2v','.w2v','.ft']
report = []
for ext in extentions:
    report.append(ext)
    for name in names:
        #report.append(name)
        dataset = pickle.load(open(datasetsDir + name+ext, 'rb'))
        #dataset = pd.concat([dataset['test'],dataset['train']],ignore_index=True)
        '''print(name)
        print(ext)
        print(dataset['classes'])'''
        meanScore = CrossClassify(dataset)
        report.append(meanScore)
print('\n'.join(report))

