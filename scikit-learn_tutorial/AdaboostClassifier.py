import pandas as pd
import numpy as np
import os

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)

print('_' * 30)
print('X type is {Xtype}'.format(Xtype = type(X)))
print('_' * 30)
print('X type is {Xtype}'.format(Xtype = type(X)))

Xdf = pd.DataFrame(X)
print('Xdf head()---------------------------')
print(Xdf.head())
ydf = pd.DataFrame(y)
print('ydf head()---------------------------')
print(ydf.head())


clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
print clf.feature_importances_

print clf.predict([[0, 0, 0, 0]])

print clf.score(X, y)


dfclf = AdaBoostClassifier(n_estimators=100, random_state=0)
dfclf.fit(Xdf, ydf)
print dfclf.feature_importances_

# print dfclf.predict([[0, 0, 0, 0]])

print dfclf.predict_proba(Xdf)

print dfclf.predict(Xdf)

print dfclf.score(Xdf, ydf)

print ydf.head(10)

print 'finish!'
