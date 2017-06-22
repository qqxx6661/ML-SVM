# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
data = []
with open("ATM/ATM2_test.txt") as file:
    for line in file:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens])
test_X = np.array(data)
print("测试输入为：", test_X)
clf_linear = joblib.load("model/model_ATM1.m")
test_X_result = clf_linear.predict(test_X)
with open("ATM/ATM2_test_result.txt", 'w') as file2:
    for line in test_X_result:
        file2.write(line)
        file2.write('\n')

# 读取成功率和时延作图
data_plot = []
y_plot = []
with open("ATM/ATM2.txt") as file:
    for line in file:
        tokens = line.strip().split(' ')
        data_plot.append([float(tk) for tk in tokens[1:3]])
        y_plot.append(tokens[-1])
plot_X = np.array(data_plot)
plot_y = np.array(y_plot)
print("画图输入为：", type(plot_X), plot_X)
print("画图输出为：", type(plot_y), plot_y)
"""
h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = plot_X[:, 0].min() - 1, plot_X[:, 0].max() + 1
y_min, y_max = plot_X[:, 1].min() - 1, plot_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
title = ['SVC with linear kernel']

for i, clf in enumerate((clf_linear),):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(plot_X[:, 0], plot_X[:, 1], c=plot_y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
plt.show()
"""