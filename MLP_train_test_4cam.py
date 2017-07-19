import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
test_num = 10000
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


sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)



start = time.time()
# 训练模型，限制树的最大深度4
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X, y)

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

test_X = sc.transform(test_X)

test_X_result = mlp.predict(test_X)
print("神经网络预测准确率：", judge_accuracy(test_X_result, test_Y))
# 91.5会变动%