#!/usr/bin/env python3
# coding=utf-8
import cv2
import datetime
import numpy as np
from sklearn.externals import joblib
# 待修改：东西出去时速度不稳定


def entropy(band):  # 计算画面熵
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()


def process_rgb_delta(cur_frame_inner, entropy_last_inner):  # 计算熵抖动
    b, g, r = cv2.split(cur_frame_inner)
    rgb_average = (entropy(r)+entropy(g)+entropy(b))/3
    if entropy_last_inner == 0:
        jitter = 0
        return rgb_average, jitter
    jitter = abs(rgb_average - entropy_last_inner)
    jitter = int(jitter)
    return rgb_average, jitter


def judge_move(cur_frame_inner, pre_frame_inner):
    gray_img = cv2.cvtColor(cur_frame_inner, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (500, 500))  # 此条不知是否影响判断
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)
    true_judge_inner = 0
    if pre_frame_inner is None:
        pre_frame_inner = gray_img
        return pre_frame_inner, true_judge_inner
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
                true_judge_inner = 1
                break
        pre_frame_inner = gray_img
        return pre_frame_inner, true_judge_inner


def cal_speed(cur_frame_inner, point_x_inner, point_y_inner):
    ok, bbox = tracker.update(cur_frame_inner)
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(cur_frame_inner, p1, p2, (0, 0, 255))  # tuple
        if point_x_inner == 0 and point_y_inner == 0:  # 刚标记后第一帧
            # 两个速度为0
            speed_x_inner = 0
            speed_y_inner = 0
        else:
            v_updown = point_y_inner - p1[1]
            v_leftright = p1[0] - point_x_inner
            speed_x_inner = v_leftright
            speed_y_inner = v_updown
        point_x_inner = p1[0]
        point_y_inner = p1[1]

        # 1号摄像头（中间）
        if p1[0] <= 50:
            open_1_inner = 1
            print("右边该开了")
        else:
            open_1_inner = 0
        open_2_inner = 1
        if p2[0] >= 590:
            open_3_inner = 1
            print("右边该开了")
        else:
            open_3_inner = 0

        return point_x_inner, point_y_inner, speed_x_inner, speed_y_inner, open_1_inner, open_2_inner, open_3_inner
    else:
        point_x_inner, point_y_inner = 9999, 9999
        # 两个速度为0
        speed_x_inner = 0
        speed_y_inner = 0
        open_1_inner = 0
        open_2_inner = 0
        open_3_inner = 0
        return point_x_inner, point_y_inner, speed_x_inner, speed_y_inner, open_1_inner, open_2_inner, open_3_inner

if __name__ == "__main__":

    # 全局变量
    fps = 29  # 获取参数一：帧率(暂时没用)
    pre_frame = None  # 获取参数一：前一帧图像（灰度），判断是否有运动物体
    entropy_last = 0  # 获取参数二：前一帧抖动数值
    flag = 0  # 是否选择跟踪目标
    point_x, point_y = 0, 0  # 获取参数三：初始化运动点
    tracker = cv2.Tracker_create("KCF")  # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN

    # 视频输入：文件或摄像头
    # camera = cv2.VideoCapture("video/sample2_1.mp4")
    camera = cv2.VideoCapture(0)
    if camera is None:
        print('请先连接摄像头或视频')
        exit()

    while True:
        res, cur_frame = camera.read()
        if res is not True:
            break


        # 跳帧参数设置，读取视频用
        '''
        totalFrameNumber = camera.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Total frame number:", totalFrameNumber)
        frame_now = 0
        frame_step = 10  # 跳过n帧运算
        frame_stop = totalFrameNumber - 1
        '''

        true_judge = 0
        jitter_now = 0
        speed_x, speed_y = 0, 0
        open_1, open_2, open_3 = 0, 0, 0


        # 读取并播放视频
        # start = time.time()
        # camera.set(cv2.CAP_PROP_POS_FRAMES, frame_now)  # 视频跳帧处理用
        res, cur_frame = camera.read()
        if res is not True:
            break
        # end = time.time()
        # seconds = end - start
        # if seconds < 1.0 / fps:
        #    time.sleep(1.0 / fps - seconds)

        # 参数0：时间(暂时不加入)
        # time_now = str(datetime.datetime.now().strftime("%H%M%S%f"))
        # row.append(time_now[:-4])  # 毫秒只取两位
        # 获取参数一：开/关
        pre_frame, true_judge = judge_move(cur_frame, pre_frame)
        # 获取参数二：图像抖动
        entropy_last, jitter_now = process_rgb_delta(cur_frame, entropy_last)

        # 检测目前是否已进入跟踪，未跟踪只获取三个参数

        if flag == 1:

            # 获取参数三：速度
            point_x, point_y, speed_x, speed_y, open_1, open_2, open_3 = cal_speed(cur_frame, point_x, point_y)
            if point_x == 9999 and point_y == 9999:
                flag = 0
                cv2.imshow('monitor', cur_frame)
                continue  # 解决输出空一行问题

        if flag == 0:

            if cv2.waitKey(20) & 0xFF == ord("s"):
                bbox = cv2.selectROI(cur_frame, False)
                ok = tracker.init(cur_frame, bbox)
                flag = 1

        print(jitter_now, speed_x, speed_y, open_1, open_2, open_3)
        clf = joblib.load("model/model_rbf.m")
        test_data = (jitter_now, point_x, point_y, open_1, open_2, open_3)
        test_data = np.array(test_data)
        predict_judge = clf.predict([test_data])
        print(true_judge, predict_judge[0])
        if true_judge == int(predict_judge[0]):
            print("判断正确")
        print('---------------')

        cv2.imshow('real time', cur_frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break

        # 跳帧处理，读取视频用
        '''
        frame_now = frame_now + frame_step
        if frame_now > frame_stop:
            break
        '''

    camera.release()
    cv2.destroyAllWindows()
