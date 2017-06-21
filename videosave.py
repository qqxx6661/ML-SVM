#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import cv2

vc = cv2.VideoCapture('video/final.mp4') #读入视频文件

if vc.isOpened(): #判断是否正常打开
    totalFrameNumber = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print(totalFrameNumber)

else:
    rval = False
    print(rval)

frametostart=0
vc.set(cv2.CAP_PROP_POS_FRAMES,frametostart)

frametostop=totalFrameNumber-1
c=frametostart
unstop=1

while unstop:
    rval ,frame=vc.read()
    cv2.imwrite('image/'+str(c) + '.jpg',frame)
    c=c+1
    if c>frametostop:
        unstop=0


vc.release()
cv2.destroyAllWindows()