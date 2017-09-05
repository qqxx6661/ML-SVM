#!/usr/bin/env python3
# coding=utf-8
import cv2
import datetime
import numpy as np
import csv
import time


def entropy(band):  # 计算画面熵
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()


def process_rgb_delta(cur_frame_inner, entropy_last_inner):  # 计算熵抖动
    b, g, r = cv2.split(cur_frame_inner)
    rgb_average = (entropy(r)+entropy(g)+entropy(b))/3
    if entropy_last_inner == 0:
        row.append(0)
        return rgb_average
    jitter = abs(rgb_average - entropy_last_inner)
    jitter = int(jitter)
    # print("画面抖动数值:", jitter)
    row.append(jitter)
    return rgb_average


def judge_move(cur_frame_inner, pre_frame_inner):
    gray_img = cv2.cvtColor(cur_frame_inner, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (500, 500))  # 此条不知是否影响判断
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)
    if pre_frame_inner is None:
        pre_frame_inner = gray_img
        return pre_frame_inner
    else:
        img_delta = cv2.absdiff(pre_frame_inner, gray_img)
        thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 500:  # 设置敏感度
                continue
            else:
                # print(cv2.contourArea(c))
                # print("画面中有运动物体")
                row.pop()
                row.append('1')
                break
        pre_frame_inner = gray_img
        return pre_frame_inner


def cal_speed(cur_frame_inner, point_x_inner, point_y_inner, tracker_inner, cam_id_inner):  # x,y速度，012开否
    ok, bbox = tracker_inner.update(cur_frame_inner)
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(cur_frame_inner, p1, p2, (0, 0, 255))  # tuple,p1是左上角,p2是右下角
        if point_x_inner == 0 and point_y_inner == 0:  # 刚标记后第一帧
            # 两个速度为0
            row.append(0)
            row.append(0)
        else:
            v_updown = point_y_inner - p1[1]
            v_leftright = p1[0] - point_x_inner
            # print("横轴速度为：", v_leftright)
            # print("纵轴速度为：", v_updown)
            row.append(v_leftright)
            row.append(v_updown)
        point_x_inner = p1[0]
        point_y_inner = p1[1]

        # 0号摄像头（左边）
        if cam_id_inner == 0:
            row.append(100)
            if p2[0] >= 590:
                if p2[0] > 680:
                    row.append(0)
                    print("右边全部出去，填充0")  # 防止一直有框
                elif p2[0] > 640:
                    row.append(50)
                    print("右边开始出去了", 50)
                else:
                    row.append(p2[0]-590)
                    print("右边该开了", p2[0]-590)
            else:
                row.append(0)
        # 1号摄像头（右边）
        else:
            if p1[0] <= 50:
                if p1[0] < 0:
                    row.append(50)
                else:
                    row.append(50-p1[0])
                print("左边该开了", 50-p1[0])
            else:
                row.append(0)
            row.append(100)

        return point_x_inner, point_y_inner
    else:
        point_x_inner, point_y_inner = 9999, 9999
        # 两个速度为0
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)
        return point_x_inner, point_y_inner


def cal_speed_upperbody(cur_frame_inner, point_x_inner, point_y_inner, cam_id_inner):

    gray = cv2.cvtColor(cur_frame_inner, cv2.COLOR_BGR2GRAY)

    bodys = bodyCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # 越小越慢，越可能检测到
        minNeighbors=2,  # 越小越慢，越可能检测到
        minSize=(95, 80),
        maxSize=(150, 180),
        # minSize=(30, 30)
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(bodys) == 0:  # 没有人脸，速度和摄像头都为0
        # 一个自身判断无运动，两个速度，两个摄像头
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)
        row.append(0)

    else:

        row.append(1)  # 自身判断有运动
        # 只输入第一张人脸数据
        print('Now face:', bodys)
        x, y, w, h = bodys[0][0], bodys[0][1], bodys[0][2], bodys[0][3]
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(cur_frame_inner, p1, p2, (0, 255, 0), 2)

        if point_x_inner == 0 and point_y_inner == 0:  # 刚标记后第一帧
            # 两个速度为0
            row.append(0)
            row.append(0)
        else:
            v_updown = point_y_inner - p1[1]
            v_leftright = p1[0] - point_x_inner
            # print("横轴速度为：", v_leftright)
            # print("纵轴速度为：", v_updown)
            row.append(v_leftright)
            row.append(v_updown)

        point_x_inner = p1[0]
        point_y_inner = p1[1]

        # 0号摄像头（左边）
        if cam_id_inner == 0:
            row.append(100)
            if p2[0] >= 590:
                if p2[0] > 680:
                    row.append(0)
                    print("右边全部出去，填充0")  # 防止一直有框
                elif p2[0] > 640:
                    row.append(50)
                    print("右边开始出去了", 50)
                else:
                    row.append(p2[0]-590)
                    print("右边该开了", p2[0]-590)
            else:
                row.append(0)
        # 1号摄像头（右边）
        else:
            if p1[0] <= 50:
                if p1[0] < 0:
                    row.append(50)
                else:
                    row.append(50-p1[0])
                print("左边该开了", 50-p1[0])
            else:
                row.append(0)
            row.append(100)

    return point_x_inner, point_y_inner


if __name__ == "__main__":

    # 全局变量
    global_start = time.time()  # 计算总用时
    time_stamp = 1  # 时间标记
    fps = 29  # 获取参数一：帧率(暂时没用)
    pre_frame0, pre_frame1 = None, None  # 获取参数一：前一帧图像（灰度），判断是否有运动物体
    entropy_last0, entropy_last1 = 0, 0  # 获取参数二：前一帧抖动数值
    point_x0, point_y0, point_x1, point_y1 = 0, 0, 0, 0  # 获取参数三：初始化运动点
    cascPath = "Webcam-Face-Detect/haarcascade_upperbody.xml"
    bodyCascade = cv2.CascadeClassifier(cascPath)

    # 视频输入：文件或摄像头
    camera0 = cv2.VideoCapture("2017-08-07 17-54-50_0.avi")
    camera1 = cv2.VideoCapture("2017-08-07 17-54-50_1.avi")

    # 打开csv文件逐行写入
    row = []
    file_name = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    with open('train_2cam/data_' + file_name + '.csv', 'w', newline='') as f:  # newline不多空行
        f_csv = csv.writer(f)

        # 循环获取参数
        while True:
            res0, cur_frame0 = camera0.read()
            res1, cur_frame1 = camera1.read()
            if res0 is not True:
                break
            if res1 is not True:
                break
            if cv2.waitKey(10) & 0xFF == 27:
                break

            cam_id = 0

            # 参数0：时间(暂时不加入)
            # time_now = str(datetime.datetime.now().strftime("%H%M%S%f"))
            # row.append(time_now[:-4])  # 毫秒只取两位
            row.append(time_stamp)
            time_stamp += 1
            print('------', time_stamp, '-------')

            # 获取参数一：开/关
            row.append('0')  # 判断有无运动，遇到有物体运动再改为1
            pre_frame0 = judge_move(cur_frame0, pre_frame0)

            # 获取参数二：图像抖动
            entropy_last0 = process_rgb_delta(cur_frame0, entropy_last0)

            # 获取参数三：速度和对应摄像头开关
            point_x0, point_y0 = cal_speed_upperbody(cur_frame0, point_x0, point_y0, cam_id)



            cam_id = 1

            # 获取参数一：开/关
            row.append('0')  # 遇到有物体运动再改为1
            pre_frame1 = judge_move(cur_frame1, pre_frame1)

            # 获取参数二：图像抖动
            entropy_last1 = process_rgb_delta(cur_frame1, entropy_last1)

            # 获取参数三：速度和对应摄像头开关
            point_x1, point_y1 = cal_speed_upperbody(cur_frame1, point_x1, point_y1, cam_id)
            # 写入一行
            # f_csv.writerow(row)
            # row = []

            # 写入一行
            print(type(row), row)
            f_csv.writerow(row)
            row = []

            cv2.imshow('frame0', cur_frame0)
            cv2.imshow('frame1', cur_frame1)


    # 计算总用时，释放内存
    global_end = time.time()
    print("global_time:", global_end - global_start)
    camera0.release()
    camera1.release()
    cv2.destroyAllWindows()
