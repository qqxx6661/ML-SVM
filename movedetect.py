#!/usr/bin/env python3
# coding=utf-8
import cv2
import time
import numpy as np
import csv


def entropy(band):
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()


def process_rgb_delta(cur_frame_inner, entropy_last_inner):
    b, g, r = cv2.split(cur_frame_inner)
    rgb_average = (entropy(r)+entropy(g)+entropy(b))/3
    jitter = abs(rgb_average - entropy_last_inner)
    print("Entropy Jitter:", jitter)
    return rgb_average, jitter


if __name__ == "__main__":

    # 全局变量
    global_start = time.time()
    fps = 29  # 获取参数一：帧率(暂时没用)
    pre_frame = None  # 获取参数一：前一帧图像（灰度）
    entropy_last = 0  # 获取参数二：前一帧抖动数值

    # 视频输入：文件或摄像头
    camera = cv2.VideoCapture(0)
    if camera is None:
        print('请先连接摄像头或视频')
        exit()

    # 跳帧参数设置，读取视频用
    '''
    totalFrameNumber = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frame number:", totalFrameNumber)
    frame_now = 0
    frame_step = 10  # 跳过n帧运算
    frame_stop = totalFrameNumber - 1
    '''

    # 打开csv文件逐行写入
    headers = ['time', 'status', 'jitter']
    row = []
    file_name = str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
    with open('data_' + file_name + '.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)

        # 循环获取参数
        while True:
            # 读取并播放视频
            # 摄像头可能需要用
            # start = time.time()
            # camera.set(cv2.CAP_PROP_POS_FRAMES, frame_now)  # 读取视频用
            res, cur_frame = camera.read()
            if res is not True:
                break
            # end = time.time()
            # seconds = end - start
            # if seconds < 1.0 / fps:
            #    time.sleep(1.0 / fps - seconds)
            cv2.imshow('monitor', cur_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            row.append(time.strftime("%H%M%S", time.localtime()))
            row.append('0')  # 遇到有物体运动再改为1
            # 获取参数一：开/关
            gray_img = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img, (500, 500))  # 此条不知是否影响判断
            gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)
            if pre_frame is None:
                pre_frame = gray_img
            else:
                img_delta = cv2.absdiff(pre_frame, gray_img)
                thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 500:  # 设置敏感度
                        continue
                    else:
                        # print(cv2.contourArea(c))
                        print("画面中有运动物体")
                        row.pop()
                        row.append('1')
                        break
                pre_frame = gray_img

            # 获取参数二：图像抖动
            entropy_now, jitter = process_rgb_delta(cur_frame, entropy_last)
            entropy_last = entropy_now
            row.append(jitter)

            # 写入一行
            f_csv.writerow(row)
            row = []

            # 跳帧处理，读取视频用
            '''
            frame_now = frame_now + frame_step
            if frame_now > frame_stop:
                break
            '''

    # 计算总用时，释放内存
    global_end = time.time()
    print("global_time:", global_end - global_start)
    camera.release()
    cv2.destroyAllWindows()
