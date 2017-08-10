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
    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            # print(predict_array[i], real_array[i])
            correct += 1
    correct_rate = correct / len(predict_array)
    return correct_rate


def execute_model():
    data = []
    labels = []
    with open("test/test_2cam_scene2(1)_901.csv") as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[1:11]])
            labels.append(tokens[0])
    test_X = np.array(data)
    test_Y = np.array(labels)
    # print("测试输入为：", test_X)
    clf_linear = joblib.load("model_2cam/model_linear.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("linear预测准确率：", judge_accuracy(test_X_result, test_Y))
    print("linear预测准确率2：", judge_accuracy_ave(test_X_result, test_Y))

    clf_linear = joblib.load("model_2cam/model_rbf.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("rbf预测准确率：", judge_accuracy(test_X_result, test_Y))
    print("rbf预测准确率2：", judge_accuracy_ave(test_X_result, test_Y))
    '''
    clf_linear = joblib.load("model_2cam/model_poly.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("poly预测准确率：", judge_accuracy(test_X_result, test_Y))
    '''
    clf_linear = joblib.load("model_2cam/model_sigmoid.m")
    test_X_result = clf_linear.predict(test_X)
    # print("预测结果：", test_X_result)
    # print("正确结果：", test_Y)
    print("sigmoid预测准确率：", judge_accuracy(test_X_result, test_Y))
    print("sigmoid预测准确率2：", judge_accuracy_ave(test_X_result, test_Y))

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
    observer.schedule(event_handler, "D:/Github/ML-SVM/model_2cam", True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

