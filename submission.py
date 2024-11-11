from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
import joblib

############################ question 1-1 ############################

### This code shows how to use KFold to do cross_validation.
### This is just one of many ways to manage training and test sets in sklearn.

wine = load_wine()
X, y = wine.data, wine.target
scores = []
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X) :
    X_train, X_test, y_train, y_test = \
        (X[train_index], X[test_index], y[train_index], y[test_index])
    clf = tree.DecisionTreeClassifier() # create empty tree
    clf.fit(X_train, y_train) # training
    scores.append(clf.score(X_test, y_test)) # predict, accuracy 등

print ("[ question #1-1 ]")
print(scores)
print()