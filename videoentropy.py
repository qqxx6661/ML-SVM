#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import cv2
import time
from PIL import Image


def entropy(band):
    hist, _ = np.histogram(band, bins=range(0, 256))
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()


def show_entropy(band_name, band):
    bits = entropy(band)
    per_pixel = bits / band.size
    print("{:3s} entropy = {:.2f} bits, {:.6f} per pixel"
          .format(band_name, bits, per_pixel))


def process(img_file):
    im = Image.open(img_file)
    print(img_file, im.format, im.size, im.mode)
    print()

    rgb = im.convert("RGB")
    hsv = im.convert("HSV")
    grey = im.convert("L")
    r, g, b = [np.asarray(component) for component in rgb.split()]
    h, s, v = [np.asarray(component) for component in hsv.split()]
    l = np.asarray(grey)

    show_entropy("R", r)
    show_entropy("G", g)
    show_entropy("B", b)
    print()

    show_entropy("H", h)
    show_entropy("S", s)
    show_entropy("V", v)
    print()

    show_entropy("L", l)
    print()


def process_rgb_average(img_file):
    im = Image.open(img_file)
    print("Processing image:",img_file)
    rgb = im.convert("RGB")
    r, g, b = [np.asarray(component) for component in rgb.split()]
    RGB_average = (entropy(r)+entropy(g)+entropy(b))/3
    print("Entropy:", RGB_average)


def process_rgb_delta(img_file, entropy_last_inner):
    # im = Image.open(img_file)  # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    # print("Processing image:", img_file)
    # rgb = im.convert("RGB")  # <class 'PIL.Image.Image'>
    # r, g, b = [np.asarray(component) for component in rgb.split()] # <PIL.Image.Image image mode=L size=1280x720 at>
    b, g, r = cv2.split(frame)
    rgb_average = (entropy(r)+entropy(g)+entropy(b))/3
    jitter = abs(rgb_average - entropy_last_inner)
    print("Entropy Jitter:", jitter)
    return rgb_average

if __name__ == "__main__":

    start = time.time()
    rval = True
    totalFrameNumber = 0
    # 读入视频文件
    vc = cv2.VideoCapture('video/final.mp4')
    # 判断是否正常打开
    if vc.isOpened():
        totalFrameNumber = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        # 输出视频总帧数
        print("Total frame number:", totalFrameNumber)
    else:
        rval = False
        print(rval)
    # 开始，结束帧
    frame_start = 0
    frame_stop = totalFrameNumber - 1
    frame_now = frame_start
    entropy_now = 0
    while rval:
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
        rval, frame = vc.read()  # frame: numpy.ndarray
        print(rval)
        # 显示视频，按Q退出
        cv2.imshow("frame", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        entropy_now = process_rgb_delta(frame, entropy_now)
        # 每n秒获取一帧
        frame_now = frame_now + 29
        if frame_now > frame_stop:
            rval = False

    end = time.time()
    print("time:", end - start)
    vc.release()
    cv2.destroyAllWindows()
