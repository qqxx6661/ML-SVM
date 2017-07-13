import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier


def judge_accuracy(predict_array, real_array):
    correct = 0
    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            # print(predict_array[i], real_array[i])
            correct += 1
    correct_rate = correct / len(predict_array)
    return correct_rate

data = []
labels = []
test_num = 2000
with open("train_4cam/train.csv") as file:
    for line in file:
        tokens = line.strip().split(',')
        data.append([tk for tk in tokens[1:28]])
        # print(data)
        labels.append(tokens[0])
print('输入参数个数：', len(data[0]))
if len(data) > test_num:  # 控制读取行数
    data = data[-test_num:]
    labels = labels[-test_num:]

X = np.array(data)
y = np.array(labels)

print("读取输入样本数为：", len(X))
# print(X)
print("读取标记样本数为：", len(y))
# print(y)
start = time.time()
# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=5)
# 拟合模型
clf.fit(X, y)
end = time.time()
print("训练模型:", end - start)

data = []
labels = []
with open("test/test_4cam.csv") as file:
    for line in file:
        tokens = line.strip().split(',')
        data.append([tk for tk in tokens[1:28]])
        labels.append(tokens[0])
test_X = np.array(data)
test_Y = np.array(labels)
test_X_result = clf.predict(test_X)

print("决策树预测准确率：", judge_accuracy(test_X_result, test_Y))

# 91.01%