# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC

data = []
labels = []
with open("1.txt") as file:
    for line in file:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])

X = np.array(data)
y = np.array(labels)

clf = SVC().fit(X, y)
print(clf.predict([[2.2, 80], [1.9, 65]]))
