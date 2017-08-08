# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import time
from watchdog.events import *
from watchdog.observers import Observer
import csv

# 检测创建


class FileEventHandler(FileSystemEventHandler):
    flag = 1
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
            print(event.src_path[-8:])
            if event.src_path[-8:] == 'rain.csv':
                pass
            else:
                self.flag += 1
            print('目前游标：', self.flag)
            if self.flag == 2:
                time.sleep(5)  # 防止读取错误
                add_train(event.src_path)
                execute_train()
                self.flag = 0


def add_train(file_src):
    data_each = []
    data = []
    labels = []
    print("读取文件：", file_src)
    with open(file_src) as file:
        for line in file:
            tokens = line.strip().split(',')
            for tk in tokens[1:6]:
                data_each.append(tk)
            for tk in tokens[7:12]:
                data_each.append(tk)
            data.append(data_each)
            # print(len(data_each))
            data_each = []
            if tokens[0] == '0' and tokens[6] == '0':
                labels.append(0)
            if tokens[0] == '1' and tokens[6] == '0':
                labels.append(1)
            if tokens[0] == '0' and tokens[6] == '1':
                labels.append(2)
            if tokens[0] == '1' and tokens[6] == '1':
                labels.append(3)

    row = []
    with open('train_2cam/train.csv', 'a', newline='') as f:  # newline不多空行, a是追加模式
        f_csv = csv.writer(f)
        #print(len(data), len(labels))
        #print(data[0])
        for i in range(len(data)):
            row.append(labels[i])
            for j in range(len(data[0])):
                row.append(data[i][j])
            f_csv.writerow(row)
            row = []


def execute_train():
    data = []
    labels = []
    test_num = 10000
    with open("train_2cam/train_4505.csv") as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[1:13]])
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

    print("进行linear训练")
    start = time.time()
    clf_linear = SVC(kernel='linear').fit(X, y)
    joblib.dump(clf_linear, "model_2cam/model_linear.m")
    # print("预测结果为：", clf_linear.predict([[91	, 93.41, 624.14], [95, 95.66, 546.82]]))
    end = time.time()
    print("linear_time:", end - start)

    print("进行rbf训练")
    start = time.time()
    clf_rbf = SVC().fit(X, y)
    joblib.dump(clf_rbf, "model_2cam/model_rbf.m")
    end = time.time()
    print("rbf_time:", end - start)
    '''
    print("进行poly训练")
    start = time.time()
    clf_poly = SVC(kernel='poly', degree=3).fit(X, y)
    joblib.dump(clf_poly, "model_2cam/model_poly.m")
    end = time.time()
    print("poly_time:", end - start)
    '''
    print("进行sigmoid训练")
    start = time.time()
    clf_sigmoid = SVC(kernel='sigmoid').fit(X, y)
    joblib.dump(clf_sigmoid, "model_2cam/model_sigmoid.m")
    end = time.time()
    print("sigmoid_time:", end - start)


if __name__ == "__main__":
    execute_train()
    observer = Observer()
    event_handler = FileEventHandler()
    observer.schedule(event_handler, "D:/Github/ML-SVM/train_2cam", True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
