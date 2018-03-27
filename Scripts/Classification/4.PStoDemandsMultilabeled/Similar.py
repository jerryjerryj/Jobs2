import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

def GetDistance(v1,v2):
    squaresSums = 0
    for i in range(0, v1.__len__()):
        squaresSums+=(v2[0]-v1[0])**2
    return squaresSums**0.5

def PredictClass(X_train,y_train, vector):
    lessDistance = GetDistance(X_train[0],vector)
    bestVectorIndex = 0
    for i in range(1,X_train.__len__()):
        distance = GetDistance(X_train[i],vector)
        if distance<lessDistance:
            lessDistance = distance
            bestVectorIndex = i
    return y_train[bestVectorIndex]

def Predict(X_train, X_test, y_train):
    result =  [PredictClass(X_train,y_train,x) for x in X_test]
    result = np.asarray(result)
    return result


class SimpleClassifier:
    class Pairs:
        def __init__(self, key):
            self.Key = key
            self.Vectors = pd.DataFrame()
            self.Index = 0
        def AddVector(self,vector):
            self.Vectors[self.Index] = vector
            self.Index+=1
        def GetAverageVector(self):
            vectors = self.Vectors.transpose()
            return vectors.mean().values.tolist()

    def __init__(self):
        self.vectors = []
        self.labels = []

    def Train(self,X_train, y_train):
        y_train = y_train.tolist()
        grouped = []
        for i in range(0, y_train.__len__()):
            key = y_train[i]
            if key not in grouped.keys():
                grouped[key] = []
            grouped[key].append(X_train[i])
        print(grouped)


datasetsDir ='F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets'
names = ['\\Администратор БД',
         '\\Программист',
         '\\Разработчик Web и мультимедийных приложений',
         '\\Системный администратор ИКТ систем',
         '\\Специалист по большим данным',
         '\\Специалист по интернет-маркетингу',
         '\\Специалист по информационным ресурсам',
         '\\Специалист по тестированию в области информационных технологий']

for name in names[0:1]:
#name = names[0]

    dataset = pickle.load(open(datasetsDir + name+'.w2v', 'rb'))
    dataset = pd.concat([dataset['test'],dataset['train']],ignore_index=True)
    vectors = dataset['vectors'].tolist()
    classes = MultiLabelBinarizer().fit_transform(dataset['classes'])
    X_train, X_test, y_train, y_test = train_test_split(vectors,classes,test_size=0.5)

    classifier = SimpleClassifier()
    classifier.Train(X_train,y_train)
    '''predictions = Predict(X_train, X_test, y_train)
    report = classification_report(y_test, predictions)
    print(name)
    print(report)'''
