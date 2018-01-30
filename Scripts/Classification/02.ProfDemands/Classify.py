from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from  sklearn.ensemble import  GradientBoostingClassifier
from sklearn import svm
from sklearn.tree import  DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics.classification import log_loss
import pickle
import numpy as np

def ClassificationHandler(pred_probs, threshold):
    predictions = []
    for pred_prob in pred_probs:
        pred_class = -1
        max = pred_prob.max()
        if max >threshold:
            pred_class = pred_prob.argmax()
        predictions.append(pred_class)
    return predictions

profName = '0Администратор БД'

with open(profName+'.train.dataset','rb') as f:
    train = pickle.load(f)
    X_train = train['vectors'].tolist()
    y_train = train['classes'].tolist()
with open(profName+'.test.dataset','rb') as f:
    test = pickle.load(f)
    X_test = test['vectors'].tolist()
    y_test = test['classes'].tolist()


model =LogisticRegression()
model.fit(X_train,y_train)
pickle.dump(model, open('bestModelGaussian.p', "wb" ) )

y_pred = model.predict(X_test)
report = classification_report(y_test,y_pred)
print(report)


y_pred_proba = model.predict_proba(X_test)
#print(y_pred_proba[0:10])
#print(log_loss(y_test,y_pred))

y_pred_handled = ClassificationHandler(y_pred_proba, .5)
#print(y_test)
#print(y_pred_handled)

report = classification_report(y_test,y_pred_handled)
print(report)
