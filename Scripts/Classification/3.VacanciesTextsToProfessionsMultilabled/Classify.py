import pickle
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from Scripts.Classification.Plots import plot_confusion_matrix


def ClassifyCross(pathToDataset, name):
    dataset = pickle.load(open(pathToDataset, 'rb'))
    cl = dataset['classes']
    classes = MultiLabelBinarizer().fit_transform(dataset['classes'])
    vectors = dataset['vectors'].tolist()

    ovr = OneVsRestClassifier(LogisticRegression())
    scores = cross_val_score(ovr, vectors, classes, cv=5,   scoring='f1_macro')
    print(name)
    print(scores)
    print('Average F1-macro: '+str(scores.mean())+'\n')

    #ensemble_jaccard_score = jaccard_similarity_score(Y_test, Y_pred_ovr)
    #print(ensemble_jaccard_score)
    #report = classification_report(Y_test, Y_pred_ovr)
    #print(report)
    #pickle.dump(ovr, open(modelName+'.p', 'wb'))

    '''X_train, X_test, Y_train, Y_test = train_test_split(vectors, classes, test_size=0.3)
    ovr.fit(X_train, Y_train)
    Y_pred_ovr = ovr.predict(X_test)
    report = classification_report(Y_test, Y_pred_ovr)
    print(Y_pred_ovr)
    print(report)'''

W2VDataset = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DataSets\W2V.dataset'
W2VTFIDFDataset = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DataSets\W2VTfIdf.dataset'
TFIDFDataset = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DataSets\TfIdf.dataset'

ClassifyCross(W2VDataset,'Word2Vec')
#ClassifyCross(TFIDFDataset,'TF-IDF')
ClassifyCross(W2VTFIDFDataset,'Word2Vec & TF-IDF')



'''cm = confusion_matrix(Y_test,Y_pred_ovr)
plot_confusion_matrix(cm,[
"Администратор БД",
"Программист",
"Разработчик Веб и мультимедиа",
"Руководитель проектов в области ИТ (Project manager)",
"Системный администратор",
"Системный аналитик",
"Специалист по большим данным",
"Специалист по защите информации в автоматизированных системах",
"Специалист по интернет-маркетингу",
"Специалист по информационным ресурсам",
"Специалист по тестированию",
"Технический писатель"])'''
