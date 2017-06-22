# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
import time
data = []
labels = []
with open("ATM/ATM1.txt") as file:
    for line in file:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])

X = np.array(data)
y = np.array(labels)

print(X)
print(y)
"""
start = time.time()
clf = SVC().fit(X, y)
print(clf.predict([[89.09, 116.16], [89, 5000], [95, 500], [82, 584]]))
end = time.time()
print("time:", end - start)
"""
start = time.time()
clf_linear = SVC(kernel='linear').fit(X, y)
print("预测结果为：", clf_linear.predict([[55, 89.09, 116.16], [2, 100, 5000], [152, 95, 500], [1345, 82, 584]]))
end = time.time()
print("time:", end - start)
"""
start = time.time()
clf_poly = SVC(kernel='poly', degree=3).fit(X, y)
print(clf_poly.predict([[89.09, 116.16], [89, 5000], [95, 500], [82, 584]]))
end = time.time()
print("time:", end - start)
"""
start = time.time()
clf_sigmoid = SVC(kernel='sigmoid').fit(X, y)
print(clf_sigmoid.predict([[89.09, 116.16], [89, 5000], [95, 500], [90, 55584]]))
end = time.time()
print("time:", end - start)
