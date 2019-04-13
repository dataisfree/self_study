# -*- utf-8 -*-
'''
@purpose: 1. 测试优化器在读入Dataframe格式数据时，优化器将其会不会转换为np.array()格式数据。
           2.clf.predict_proba() 结果为概率数组，数组标签顺序是什么样，是否同标签相对应的“数字label” 排序一致。如：1, 2, 3, ...
@result: 1.优化器fit时先判断是否为np.array(),否的话，转为np.array() 后进行fit.
        2.结果数组标签顺序同“数字标签”顺序一致。如：
        array：
                    label_1     label_2
                0   0.45        0.55
                1   0.6         0.4
@method: debug || step in || read source code!
'''

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
