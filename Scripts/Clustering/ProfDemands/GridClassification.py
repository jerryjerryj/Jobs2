from Scripts.Tools.IO import IOTools
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
from Scripts.Clustering.ProfDemands.TestSplitting import GetCommonSplit, GetTrainTestSplit


def Classify(algorithm, name, X_train, X_test, y_train, y_test):
    algorithm.fit(X_train, y_train)
    prediction = algorithm.predict(X_test)
    f_score = f1_score(prediction,y_test, average="micro")
    #print(name+"\t\t"+str(accuracy)+"\t\t"+str(precision)+"\t\t"+str(recall)+"\t\t"+str(f_score)+"\t\t"+str(elapsed)+"\t\t")
    print(name+"\t\t"+str(f_score))


X_train, X_test, y_train, y_test = GetTrainTestSplit(0.3)

param_grid = {"criterion": ["gini", "entropy"],
              "class_weight" :["balanced","balanced_subsample"],
              "n_estimators" : range(50,60,10)
              }
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
Classify(grid_search,'\nRandom Forest\n(with GridSearch)', X_train, X_test, y_train, y_test)


param_grid = {
              "fit_intercept" :[True, False],
              "class_weight" : ["balanced", None],
              "solver" : ['newton-cg', 'lbfgs',  'sag'],
              'multi_class' : ['ovr', 'multinomial'],
              }
grid_search = GridSearchCV(LogisticRegression(), param_grid=param_grid)
Classify(grid_search,'\nLog Regression\n(with GridSearch)', X_train, X_test, y_train, y_test)

pickle.dump(grid_search, open("model", "wb"))
