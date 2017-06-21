#!/usr/bin/env python3
# coding=utf-8
import cv2
import time

camera = cv2.VideoCapture('video/test1.mp4')
if camera is None:
    print('请先连接摄像头或图像')
    exit()

fps = 5  # 帧率
pre_frame = None  # 总是取前一帧做为背景（不用考虑环境影响）


while True:
    # start = time.time()
    res, cur_frame = camera.read()
    if res is not True:
        break
    # end = time.time()
    # seconds = end - start
    # if seconds < 1.0 / fps:
    #     time.sleep(1.0 / fps - seconds)

    cv2.imshow('img', cur_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

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
                print("前一帧和当前帧不一样了, 有什么东西在动!")
                break

        pre_frame = gray_img

camera.release()
cv2.destroyAllWindows()