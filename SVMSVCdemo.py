# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
data = []
labels = []
with open("ATM/ATM1.txt") as file:
    for line in file:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        # print(data)
        labels.append(tokens[-1])

X = np.array(data)
y = np.array(labels)

print("读取输入为：", type(X))
print(X)
print("读取输出为：", type(y))
print(y)
start = time.time()
clf_linear = SVC(kernel='linear').fit(X, y)
joblib.dump(clf_linear, "model/model_ATM5.m")
print("预测结果为：", clf_linear.predict([[91	, 93.41, 624.14], [95, 95.66, 546.82]]))
print("预测结果为：", clf_linear.predict([[103, 95.15, 39328.5], [112, 88.55, 5000]]))
print("预测结果为：", clf_linear.predict([[76, 98.68, 53267.82], [582, 80, 99999]]))
print("预测结果为：", clf_linear.predict([[47, 100, 16256411.17], [65, 56.45, 25621]]))
end = time.time()
print("time:", end - start)

"""
start = time.time()
clf = SVC().fit(X, y)
print(clf.predict([[89.09, 116.16], [89, 5000], [95, 500], [82, 584]]))
end = time.time()
print("time:", end - start)

start = time.time()
clf_poly = SVC(kernel='poly', degree=3).fit(X, y)
print(clf_poly.predict([[89.09, 116.16], [89, 5000], [95, 500], [82, 584]]))
end = time.time()
print("time:", end - start)

start = time.time()
clf_sigmoid = SVC(kernel='sigmoid').fit(X, y)
print("预测结果为：", clf_linear.predict([[91	, 93.41, 624.14], [76, 98.68, 53267.82], [103, 95.15, 39328.5], [47, 100, 16256411.17]]))
print("预测结果为：", clf_linear.predict([[65	, 56.45, 5000]]))
end = time.time()
print("time:", end - start)
"""