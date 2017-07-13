#!/usr/bin/env python3
# coding=utf-8
import cv2
import datetime
import numpy as np
import csv
import time

# 待修改：东西出去时速度不稳定


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

        # 0号摄像头（最左边）
        if cam_id_inner == 0:
            row.append(100)
            if p2[0] >= 590:
                if p2[0] > 640:
                    row.append(50)
                else:
                    row.append(p2[0]-590)
                print("右边该开了", p2[0]-590)
            else:
                row.append(0)
            row.append(0)
            row.append(0)
        # 1号摄像头（中间）
        elif cam_id_inner == 1:
            if p1[0] <= 50:
                if p1[0] < 0:
                    row.append(50)
                else:
                    row.append(50-p1[0])
                print("左边该开了", 50-p1[0])
            else:
                row.append(0)
            row.append(100)
            if p2[0] >= 590:
                if p2[0] > 640:
                    row.append(50)
                else:
                    row.append(p2[0]-590)
                print("右边该开了", p2[0]-590)
            else:
                row.append(0)
            row.append(0)
        # 2号摄像头（中间）
        elif cam_id_inner == 2:
            row.append(0)
            if p1[0] <= 50:
                if p1[0] < 0:
                    row.append(50)
                else:
                    row.append(50-p1[0])
                print("左边该开了", 50-p1[0])
            else:
                row.append(0)
            row.append(100)
            if p2[0] >= 590:
                if p2[0] > 640:
                    row.append(50)
                else:
                    row.append(p2[0]-590)
                print("右边该开了", p2[0]-590)
            else:
                row.append(0)
        # 3号摄像头（右边）
        else:
            row.append(0)
            row.append(0)
            if p1[0] <= 50:
                if p1[0] < 0:
                    row.append(50)
                else:
                    row.append(50 - p1[0])
                print("左边该开了", 50 - p1[0])
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
        row.append(0)
        row.append(0)
        return point_x_inner, point_y_inner

if __name__ == "__main__":

    # 全局变量
    global_start = time.time()  # 计算总用时
    cam_id = 0
    fps = 29  # 获取参数一：帧率(暂时没用)
    pre_frame0, pre_frame1, pre_frame2, pre_frame3 = None, None, None, None  # 获取参数一：前一帧图像（灰度），判断是否有运动物体
    entropy_last0, entropy_last1, entropy_last2, entropy_last3 = 0, 0, 0, 0  # 获取参数二：前一帧抖动数值
    flag0, flag1, flag2, flag3 = 0, 0, 0, 0  # 是否选择跟踪目标
    point_x0, point_y0, point_x1, point_y1, point_x2, point_y2, point_x3, point_y3 = 0, 0, 0, 0, 0, 0, 0, 0  # 获取参数三：初始化运动点
    tracker0 = cv2.Tracker_create("KCF")  # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    tracker1 = cv2.Tracker_create("KCF")
    tracker2 = cv2.Tracker_create("KCF")
    tracker3 = cv2.Tracker_create("KCF")

    # 视频输入：文件或摄像头
    camera0 = cv2.VideoCapture("video/2017-07-13 13-38-12_0.avi")
    camera1 = cv2.VideoCapture("video/2017-07-13 13-38-12_1.avi")
    camera2 = cv2.VideoCapture("video/2017-07-13 13-39-06_0.avi")
    camera3 = cv2.VideoCapture("video/2017-07-13 13-39-06_1.avi")


    # 打开csv文件逐行写入
    row = []
    file_name = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    with open('train_4cam/data_' + file_name + '.csv', 'w', newline='') as f:  # newline不多空行
        f_csv = csv.writer(f)

        # 循环获取参数
        while True:
            res0, cur_frame0 = camera0.read()
            res1, cur_frame1 = camera1.read()
            res2, cur_frame2 = camera2.read()
            res3, cur_frame3 = camera3.read()
            if res0 is not True:
                break
            if res1 is not True:
                break
            if res2 is not True:
                break
            if res3 is not True:
                break

            if cv2.waitKey(10) & 0xFF == 27:
                break

            # 参数0：时间(暂时不加入)
            # time_now = str(datetime.datetime.now().strftime("%H%M%S%f"))
            # row.append(time_now[:-4])  # 毫秒只取两位

            cam_id = 3
            row.append('0')  # 遇到有物体运动再改为1
            # 获取参数一：开/关
            pre_frame0 = judge_move(cur_frame0, pre_frame0)
            # 获取参数二：图像抖动
            entropy_last0 = process_rgb_delta(cur_frame0, entropy_last0)

            # 检测目前是否已进入跟踪，未跟踪只获取三个参数
            if flag0 == 0:
                # 两个速度为0, 三个左右也写为0
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)

            if flag0 == 1:

                # 获取参数三：速度
                point_x0, point_y0 = cal_speed(cur_frame0, point_x0, point_y0, tracker0, cam_id)
                # 写入一行
                # f_csv.writerow(row)
                # row = []
                if point_x0 == 9999 and point_y0 == 9999:
                    flag0 = 0
                    # 否则目标消失时没空行
                    # f_csv.writerow(row)
                    # row = []
                    cv2.imshow('frame0', cur_frame0)
                    # continue  # 解决输出空一行问题

            cam_id = 2
            row.append('0')  # 遇到有物体运动再改为1
            # 获取参数一：开/关
            pre_frame1 = judge_move(cur_frame1, pre_frame1)
            # 获取参数二：图像抖动
            entropy_last1 = process_rgb_delta(cur_frame1, entropy_last1)

            if flag1 == 0:
                # 两个速度为0, 两个左右也写为0
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)

            if flag1 == 1:

                # 获取参数三：速度
                point_x1, point_y1 = cal_speed(cur_frame1, point_x1, point_y1, tracker1, cam_id)
                # 写入一行
                # f_csv.writerow(row)
                # row = []
                if point_x1 == 9999 and point_y1 == 9999:
                    flag1 = 0
                    # 否则目标消失时没空行
                    f_csv.writerow(row)
                    row = []
                    cv2.imshow('frame1', cur_frame1)
                    # continue  # 解决输出空一行问题

            cam_id = 1
            row.append('0')  # 遇到有物体运动再改为1
            # 获取参数一：开/关
            pre_frame2 = judge_move(cur_frame2, pre_frame2)
            # 获取参数二：图像抖动
            entropy_last2 = process_rgb_delta(cur_frame2, entropy_last2)

            if flag2 == 0:
                # 两个速度为0, 两个左右也写为0
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)

            if flag2 == 1:

                # 获取参数三：速度
                point_x2, point_y2 = cal_speed(cur_frame2, point_x2, point_y2, tracker2, cam_id)
                # 写入一行
                # f_csv.writerow(row)
                # row = []
                if point_x2 == 9999 and point_y2 == 9999:
                    flag2 = 0
                    # 否则目标消失时没空行
                    f_csv.writerow(row)
                    row = []
                    cv2.imshow('frame2', cur_frame2)
                    # continue  # 解决输出空一行问题

            cam_id = 0
            row.append('0')  # 遇到有物体运动再改为1
            # 获取参数一：开/关
            pre_frame3 = judge_move(cur_frame3, pre_frame3)
            # 获取参数二：图像抖动
            entropy_last3 = process_rgb_delta(cur_frame3, entropy_last3)

            if flag3 == 0:
                # 两个速度为0, 两个左右也写为0
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)
                row.append(0)

            if flag3 == 1:

                # 获取参数三：速度
                point_x3, point_y3 = cal_speed(cur_frame3, point_x3, point_y3, tracker3, cam_id)
                # 写入一行
                # f_csv.writerow(row)
                # row = []
                if point_x3 == 9999 and point_y3 == 9999:
                    flag3 = 0
                    # 否则目标消失时没空行
                    f_csv.writerow(row)
                    row = []
                    cv2.imshow('frame3', cur_frame3)
                    continue  # 解决输出空一行问题

            # 写入一行
            f_csv.writerow(row)
            row = []

            if cv2.waitKey(20) & 0xFF == ord("q"):
                bbox0 = cv2.selectROI(cur_frame0, False)
                ok0 = tracker0.init(cur_frame0, bbox0)
                flag0 = 1
            if cv2.waitKey(20) & 0xFF == ord("w"):
                bbox1 = cv2.selectROI(cur_frame1, False)
                ok1 = tracker1.init(cur_frame1, bbox1)
                flag1 = 1
            if cv2.waitKey(20) & 0xFF == ord("e"):
                bbox2 = cv2.selectROI(cur_frame2, False)
                ok2 = tracker2.init(cur_frame2, bbox2)
                flag2 = 1
            if cv2.waitKey(20) & 0xFF == ord("r"):
                bbox3 = cv2.selectROI(cur_frame3, False)
                ok3 = tracker3.init(cur_frame3, bbox3)
                flag3 = 1

            cv2.imshow('frame0', cur_frame0)
            cv2.imshow('frame1', cur_frame1)
            cv2.imshow('frame2', cur_frame2)
            cv2.imshow('frame3', cur_frame3)


            print('---------------')



    # 计算总用时，释放内存
    global_end = time.time()
    print("global_time:", global_end - global_start)
    camera0.release()
    camera1.release()
    cv2.destroyAllWindows()
