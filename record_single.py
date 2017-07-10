#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import cv2

cap0 = cv2.VideoCapture(2)
# 定义解码器并创建VideoWrite对象
# linux: XVID、X264; windows:DIVX
# 20.0指定一分钟的帧数
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out0 = cv2.VideoWriter('video/output0.avi', fourcc, 20.0, (640, 480))

record_status = 0
while True:
    # 读取一帧
    status0, frame0 = cap0.read()


    if status0 is not True:
        print('0 failed')
        break


    if record_status == 1:
        # frame0 = cv2.flip(frame0, 0)
        # frame1 = cv2.flip(frame1, 0) # 翻转图像
        out0.write(frame0)


    # 显示帧
    cv2.imshow('frame0', frame0)


    if cv2.waitKey(1) & 0xFF == ord('r'):
        record_status = 1
        print("Start recording")
        continue
    if cv2.waitKey(1) & 0xFF == ord('t'):
        record_status = 0
        print("Stop recording")
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放VideoCapture对象
cap0.release()
out0.release()
cv2.destroyAllWindows()

