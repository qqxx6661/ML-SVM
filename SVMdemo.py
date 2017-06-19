# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:14:16 2016

@author: ZQ
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import SVR


def plot_decision_function(X, classifier, sample_weight, axis, title):
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone)
    axis.axis('off')
    axis.set_title(title)


def loadData(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            data.append(line.strip().split('\t')[1:])
    return np.array(data[1:])


def initData(data):
    m, n = np.shape(data)
    retDat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if data[i][j] == '是':
                retDat[i][j] = 0
            elif data[i][j] == '否':
                retDat[i][j] = 1
            else:
                retDat[i][j] = float(data[i][j])
    return retDat


if __name__ == '__main__':
    data = loadData('train.txt')
    print(data)
    num_data = initData(data)
    x = num_data[:, :2]
    y = num_data[:, -1]

    """
    #线性画图
    lclf = SVC(C = 1.0,gamma=0.1,kernel='linear')
    bclf = SVC(C = 1.0,gamma = 0.1)

    plt.scatter(x[:,0],x[:,1],c = y)
    xx = np.linspace(-5,5)
    lclf.fit(x,y)
    lw = lclf.coef_[0]
    la = -lw[0]/lw[1]
    ly = la*xx - lclf.intercept_[0]/lw[1]
    h0 = plt.plot(xx,ly,'k-',label = 'linear')
    """
    """
    #高斯画图
    bclf.fit(x,y)

    weight = np.ones(len(x))
    fig,axis = plt.subplots(1,1)
    plot_decision_function(x,bclf,weight,axis,'test')
    """
    # 训练的SVR
    svr_rbf = SVR(kernel = 'rbf',C = 1e3,gamma = 3)
    X = x[:,0].reshape((17,1))
    y = x[:,1].reshape((17,1))
    y_rbf = svr_rbf.fit(X,y.ravel()).predict(X)
    plt.scatter(x[:,0],x[:,1],c = x[:,0])
    plt.scatter(x[:,0],y_rbf,c = x[:,0],marker = '*')
    plt.show()