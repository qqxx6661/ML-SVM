# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from watchdog.events import *
from watchdog.observers import Observer
import time

# 检测修改


class FileEventHandler(FileSystemEventHandler):
    flag = 0

    def __init__(self):
        FileSystemEventHandler.__init__(self)

    def on_moved(self, event):
        if event.is_directory:
            print("directory moved from {0} to {1}".format(event.src_path,event.dest_path))
        else:
            print("file moved from {0} to {1}".format(event.src_path,event.dest_path))

    def on_created(self, event):
        if event.is_directory:
            print("directory created:{0}".format(event.src_path))
        else:
            print("file created:{0}".format(event.src_path))

    def on_deleted(self, event):
        if event.is_directory:
            print("directory deleted:{0}".format(event.src_path))
        else:
            print("file deleted:{0}".format(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))
            self.flag += 1
            # print(self.flag)
            if self.flag == 2:
                execute_model()
                self.flag = 0


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
            print('precision: %s' % (M[i][i] / float(colsum)), 'recall: %s' % (M[i][i] / float(rowsum)))
        except ZeroDivisionError:
            print('precision: %s' % 0, 'recall: %s' % 0)

    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            # print(predict_array[i], real_array[i])
            correct += 1
    correct_rate = correct / len(predict_array)
    return correct_rate


def execute_model():
    data = []
    labels = []
    with open("test/test_4cam.csv") as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[1:28]])
            labels.append(tokens[0])
    test_X = np.array(data)
    test_Y = np.array(labels)
    # print("测试输入为：", test_X)
    clf_linear = joblib.load("model_4cam/model_linear.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("linear预测准确率：", judge_accuracy(test_X_result, test_Y))

    '''
    clf_linear = joblib.load("model_4cam/model_rbf.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("rbf预测准确率：", judge_accuracy(test_X_result, test_Y))
    
    clf_linear = joblib.load("model_4cam/model_poly.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("poly预测准确率：", judge_accuracy(test_X_result, test_Y))
    
    clf_linear = joblib.load("model_4cam/model_sigmoid.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("sigmoid预测准确率：", judge_accuracy(test_X_result, test_Y))
    '''


    '''
    with open("ATM/result/ATM34_test_result_linear_ATM12_time.txt", 'w') as file2:
        for line in test_X_result:
            file2.write(line)
            file2.write('\n')
    '''

if __name__ == "__main__":
    execute_model()
    observer = Observer()
    event_handler = FileEventHandler()
    observer.schedule(event_handler, "D:/Github/ML-SVM/model_4cam", True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

