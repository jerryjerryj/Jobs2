from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm,my_tags, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(my_tags))
    target_names = my_tags
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

df = pickle.load(open('DatasetFromMarked.p', "rb"))

vectors = df['vectors'].tolist()
classes = df['classes'].tolist()


X_train, X_test, Y_train, Y_test = train_test_split(vectors, classes,test_size=0.30)

model = LogisticRegression()

model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
report = classification_report(Y_test,Y_pred)
print(report)

cm = confusion_matrix(Y_test,Y_pred)
plot_confusion_matrix(cm,['Администратор БД',
'Архитектор ПО',
'Программист',
'Разработчик Web',
'Специалист по BigData',
'Специалист по интернет-маркетингу',
'Специалист по информационным ресурсам',
'Специалист по тестированию',
'Технический писатель',
'Руководитель проектов'])


