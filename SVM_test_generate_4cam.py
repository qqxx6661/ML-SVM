# -*- coding: utf-8 -*-

import csv


def add_train(file_src):
    data_each = []
    data = []
    labels = []
    print("读取文件：", file_src)
    with open(file_src) as file:
        for line in file:
            tokens = line.strip().split(',')
            for tk in tokens[1:8]:
                data_each.append(tk)
            for tk in tokens[9:16]:
                data_each.append(tk)
            for tk in tokens[17:24]:
                data_each.append(tk)
            for tk in tokens[25:32]:
                data_each.append(tk)
            data.append(data_each)
            # print(len(data_each))
            data_each = []
            if tokens[0] == '0' and tokens[8] == '0' and tokens[16] == '0' and tokens[24] == '0':
                labels.append(0)
            if tokens[0] == '1' and tokens[8] == '0' and tokens[16] == '0' and tokens[24] == '0':
                labels.append(1)
            if tokens[0] == '0' and tokens[8] == '1' and tokens[16] == '0' and tokens[24] == '0':
                labels.append(2)
            if tokens[0] == '1' and tokens[8] == '1' and tokens[16] == '0' and tokens[24] == '0':
                labels.append(3)
            if tokens[0] == '0' and tokens[8] == '0' and tokens[16] == '1' and tokens[24] == '0':
                labels.append(4)
            if tokens[0] == '0' and tokens[8] == '1' and tokens[16] == '1' and tokens[24] == '0':
                labels.append(5)
            if tokens[0] == '0' and tokens[8] == '0' and tokens[16] == '0' and tokens[24] == '1':
                labels.append(6)
            if tokens[0] == '0' and tokens[8] == '0' and tokens[16] == '1' and tokens[24] == '1':
                labels.append(7)

    row = []
    with open('test/test_4cam.csv', 'a', newline='') as f:  # newline不多空行, a是追加模式
        f_csv = csv.writer(f)
        print(len(data), len(labels))
        print(len(data[0]), data[0])
        for i in range(len(data)):
            row.append(labels[i])
            for j in range(len(data[0])):
                row.append(data[i][j])
            f_csv.writerow(row)
            row = []

if __name__ == "__main__":
    add_train('D:/Github/ML-SVM/train_4cam/data_2017-07-13 12-47-31-test.csv')
