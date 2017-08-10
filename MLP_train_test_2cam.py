import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def judge_accuracy_ave(predict_array, real_array):
    List_ave = []
    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            List_ave.append(100)
            continue
        if predict_array[i] == 0 and real_array[i] == 3:
            List_ave.append(0)
            continue
        if predict_array[i] == 3 and real_array[i] == 0:
            List_ave.append(0)
            continue
        if predict_array[i] == 2 and real_array[i] == 1:
            List_ave.append(0)
            continue
        if predict_array[i] == 1 and real_array[i] == 2:
            List_ave.append(0)
            continue
        List_ave.append(50)
    print('测试集长度：', len(List_ave))
    # print(List_ave)
    return np.mean(List_ave)




def judge_accuracy(predict_array, real_array):
    correct = 0
    M = []
    count = 0
    for classes in ['0', '1', '2', '3', '4', '5', '6', '7']:
        location = [i for i, v in enumerate(real_array) if v == classes]
        # print(type(location), len(location))
        M_each = [0, 0, 0, 0, 0, 0, 0, 0]
        for location_each in location:
            for classes_1 in ['0', '1', '2', '3', '4', '5', '6', '7']:
                if predict_array[location_each] == classes_1:
                    M_each[count] += 1
                count += 1
            count = 0
        M.append(M_each)
    print(M)
    n = len(M)
    for i in range(len(M[0])):
        rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
        try:
            print('分类', i, 'precision: %s' % (M[i][i] / float(colsum)), 'recall: %s' % (M[i][i] / float(rowsum)))
        except ZeroDivisionError:
            print('分类', i, 'precision: %s' % 0, 'recall: %s' % 0)

    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            # print(predict_array[i], real_array[i])
            correct += 1
    correct_rate = correct / len(predict_array)
    return correct_rate

data = []
labels = []
test_num = 10000
with open("train_2cam/train.csv") as file:
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
with open("test/test_2cam_scene2(1)_901.csv") as file:
    for line in file:
        tokens = line.strip().split(',')
        data.append([tk for tk in tokens[1:28]])
        labels.append(tokens[0])
test_X = np.array(data)
test_Y = np.array(labels)
test_X = sc.transform(test_X)
test_X_result = mlp.predict(test_X)
print("神经网络预测准确率：", judge_accuracy(test_X_result, test_Y))
print("神经网络预测准确率2：", judge_accuracy_ave(test_X_result, test_Y))
