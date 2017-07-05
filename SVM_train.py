# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
data = []
labels = []
with open("data_2017-07-05 14-00-01.csv") as file:
    for line in file:
        tokens = line.strip().split(',')
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
joblib.dump(clf_linear, "model/model_linear.m")
# print("预测结果为：", clf_linear.predict([[91	, 93.41, 624.14], [95, 95.66, 546.82]]))
end = time.time()
print("time:", end - start)


start = time.time()
clf_rbf = SVC().fit(X, y)
joblib.dump(clf_rbf, "model/model_rbf.m")
end = time.time()
print("time:", end - start)

start = time.time()
clf_poly = SVC(kernel='poly', degree=3).fit(X, y)
joblib.dump(clf_poly, "model/model_poly.m")
end = time.time()
print("time:", end - start)

start = time.time()
clf_sigmoid = SVC(kernel='sigmoid').fit(X, y)
joblib.dump(clf_sigmoid, "model/model_sigmoid.m")
end = time.time()
print("time:", end - start)