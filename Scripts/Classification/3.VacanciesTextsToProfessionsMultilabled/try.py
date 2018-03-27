
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata

yeast = fetch_mldata('yeast')
X = yeast['data']
Y = yeast['target'].transpose().toarray()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)
print(X[0:2])
print(Y[0:2])
# Fit an independent logistic regression model for each class using the
# OneVsRestClassifier wrapper.
ovr = OneVsRestClassifier(LogisticRegression())
ovr.fit(X_train, Y_train)
Y_pred_ovr = ovr.predict(X_test)